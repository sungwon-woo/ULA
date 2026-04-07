import gspread
import argparse
import os
import string
import math
import time
import random
import json
import subprocess

from oauth2client.service_account import ServiceAccountCredentials


_CANT_FIND_PARAMETERS = r"""
Couldn't find the cell that contained "START PARAMETER". 
Please add "START PARAMETER" to the front of the parameter column. 
"""
_LIMIT_COLUMN_ERROR = r"""
We limit the max column to ZZ.
"""
_NO_VALID_EXPERIMENT_ERROR = r"""
Couldn't find the cell that contained "END EXPERIMENT".
Please add "END EXPERIMENT" to the last row.
"""


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        return None


def process_call(command):
    # process = subprocess.Popen(command, shell=True, text=True,
    #                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = process.communicate()
    # return stdout, stderr
    with os.popen(command) as stream:
        output = stream.read()

    return output


class AutoCommand(object):
    def __init__(self, args):
        self.args = args
        self.folder_name = os.path.join(args.log_dir, args.sheet_name)
        
        # Dataset Setting
        self.end_line_number = 0
        self.param_column = []
        self.sheet_column = []
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive',
        ]
        json_file_name = 'gs_key.json'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
        self.gc = gspread.authorize(credentials)
        self.spreadsheet_url = "https://docs.google.com/spreadsheets/d/1IjYJP5trWOxrQmifQfeUa8_G8gDUQ57vXXRnbJbs3VU/edit?gid=1695959320#gid=1695959320"
        self.worksheet = self.read_sheet()
        self.params_list = self.set_parameter()

        os.makedirs(self.folder_name, exist_ok=True)

    def read_sheet(self):
        doc = self.gc.open_by_url(self.spreadsheet_url)
        return doc.worksheet(self.args.sheet_name)

    def split_arch(self, s):
        head = s.rstrip('0123456789')
        head = head.split('_')[0]
        return head

    def sheet_column_value_generator(self, column_number):
        all_column_number = column_number + 500
        loop_count = int(math.floor(all_column_number / 26))
        if loop_count >= 24:
            raise NotImplementedError(_LIMIT_COLUMN_ERROR)
        for i in range(loop_count+1):
            front_alphabet = '' if i == 0 else string.ascii_uppercase[i-1]
            last_alphabet_slice_number = 26 if i < loop_count else all_column_number % 26
            self.sheet_column.extend([front_alphabet + j for j in string.ascii_uppercase[:last_alphabet_slice_number]])
        self.sheet_column = self.sheet_column[column_number:]

    def retry_get_values_with_backoff(self, x=0, retries=5):
        while True:
            try:
                return self.worksheet.get_all_values()
            except gspread.exceptions.APIError:
                print(f"Wait in {2 ** x} sec (gspread.APIError)")
                if x > retries:
                    raise
                sleep = 2 ** x + random.uniform(0, 1)
                time.sleep(sleep)
                x += 1

    def set_parameter(self):
        lines = self.retry_get_values_with_backoff(x=0, retries=5)

        params_list = []
        param_cnt = 0
        line_cnt = 1
        for line in lines:
            # Line parameter settings
            if line[0] == "START PARAMETER":
                column_number = line.index("END PARAMETER")
                self.param_column = {i: column_name for i, column_name in enumerate(line[1:column_number])}
                self.sheet_column_value_generator(column_number)
                line_cnt += 1
                continue
            # For Write Object Count
            if line[0] == "END EXPERIMENT":
                self.end_line_number = line_cnt
                line_cnt += 1
                break
            # Skip the Null Row
            if line[0] != str(param_cnt):
                line_cnt += 1
                continue

            if len(self.param_column) == 0:
                raise NotImplementedError(_CANT_FIND_PARAMETERS)

            param_dict = {'line_num': line_cnt}

            param_dict.update({self.param_column[i]: line[i+1] for i in range(0, len(self.param_column))})
            params_list.append(param_dict)

            param_cnt += 1
            line_cnt += 1

        return params_list

    def eval_command(self):
        for param_cnt, params in enumerate(self.params_list[self.args.eval_end_point[0]:self.args.eval_end_point[1]+1]):
            command = "python train.py"
            command += " {}".format(params[self.param_column[0]])

            for i in range(1, len(self.param_column)):
                if str2bool(params[self.param_column[i]]):
                    command += " {}".format(self.param_column[i])
                    continue
                if params[self.param_column[i]].lower() == 'false':
                    continue
                if params[self.param_column[i]] != '':
                    command += " {} {}".format(self.param_column[i], params[self.param_column[i]])

            output_dir = os.path.join(self.folder_name, str(params['line_num']-2))
            command += f" -log_dir {output_dir}"

            print(command)
            #import pdb;pdb.set_trace()
            output = process_call(command)
            #print(output)
            output = output.split('\n')[-6:-1]
            output = list(map(float, output))

            # Total Score
            sheet_range = '{}:{}'.format(self.sheet_column[1] + str(params["line_num"]),
                                         self.sheet_column[len(output)+1] + str(params["line_num"]))
            cell_list = self.worksheet.range(sheet_range)
            for idx, val in enumerate(output):
                cell_list[idx].value = val
            self.worksheet.update_cells(cell_list)

    def __call__(self):
        self.eval_command()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_end_point', nargs='+', type=int, default=[0, 54])
    parser.add_argument('--sheet_name', type=str, default='configs')
    parser.add_argument('--log_dir', type=str, default='./train_log')

    args = parser.parse_args()
    random.seed(1234)

    command_module = AutoCommand(args)
    command_module()