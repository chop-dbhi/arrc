from os import listdir
from os.path import isfile, join
import string
import pandas as pd
import shutil

def first_report(file_path):
    stop_key = "END OF IMPRESSION:"
    lines = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            lines.append(line)
            if stop_key in line:
                break
    return lines

def impression(file_path, first_only=False):
    start_key = "IMPRESSION:"
    stop_key = "END OF IMPRESSION:"
    start_found = False
    stop_found = False
    stop_searching=False
    lines = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if start_found and not stop_found:
                if stop_key in line:
                    stop_found = True
                    if first_only: stop_searching = True
                else:
                    lines.append(line)
            elif start_key in line and not stop_key in line and not(stop_searching and first_only):
                lines.append(line)
                start_found = True
                stop_found = False
    return lines


def findings(file_path, first_only=False):
    start_key = "FINDINGS:"
    stop_key = "IMPRESSION:"
    start_found = False
    stop_found = False
    stop_searching = False
    lines = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if start_found and not stop_found:
                if stop_key in line:
                    stop_found = True
                    if first_only: stop_searching = True
                else:
                    lines.append(line)
            else:
                if start_key in line and not (stop_searching and first_only):
                    lines.append(line)
                    start_found = True
                    stop_found = False
    return lines

def count_findings(file_path):
    key = "FINDINGS"
    cnt = 0
    with open(file_path,'r') as f:
        for line in f.readlines():
            if key in line:
                cnt += 1
    return cnt


def load_report(path):
    f = open(path,'r')
    text = reduce(lambda x,y: x+y, f.readlines(), "")
    f.close()
    return text

if __name__ == '__main__':
    # set common path variables
    report_path = './data/input/SDS_PV2_combined/reports_single'
    label_file = './data/input/SDS_PV2_combined/SDS_PV2_class_labels.txt'
    label_data = pd.read_csv(label_file)

    root_path = './data/input/SDS_PV2_combined/'
    abnormal_multi_label_file = join(root_path,'reports_multi_abnormal','multi_abnormal_labels.txt')
    for idx, row in label_data.iterrows():
        pid = row['pid']
        report = join(report_path,'{0}.txt'.format(pid))

        if row['doc_norm'] == 1:
            if count_findings(report)>1:
                #copy the complete report to a file
                with open(join(root_path,'reports_multi_abnormal','complete','{0}.txt'.format(pid)),'a') as f:
                    f.write(string.join(first_report(report)))
                #copy the first findings/impression to a file
                fnd = string.join(findings(report,True))
                impr = string.join(impression(report, True))
                with open(join(root_path, 'reports_multi_abnormal', 'find_impr', '{0}_fi.txt'.format(pid)), 'a') as f:
                    f.write(fnd)
                    f.write("\n\n")
                    f.write(impr)
            else:
               shutil.copy(report, join(root_path, 'reports_single', '{0}.txt'.format(pid)))
               fnd = string.join(findings(report, True))
               impr = string.join(impression(report, True))
               with open(join(root_path, 'reports_single_find_impr','{0}_fi.txt'.format(pid)),'a') as f:
                   f.write(fnd)
                   f.write("\n\n")
                   f.write(impr)
        else:
            #normal doc
            #copy only the first complete report to reports_single
            with open(join(root_path,'reports_single','{0}.txt'.format(pid)),'a') as f:
                f.write(string.join(first_report(report)))
            #create findings impression file
            fnd = string.join(findings(report, True))
            impr = string.join(impression(report, True))
            with open(join(root_path, 'reports_single_find_impr','{0}_fi.txt'.format(pid)),'a') as f:
                f.write(fnd)
                f.write("\n\n")
                f.write(impr)


