from os import listdir
from os.path import isfile, join
import string

def impression(file_path):
    start_key = "IMPRESSION:"
    stop_key = "END OF IMPRESSION:"
    start_found = False
    stop_found = False
    lines = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if start_found and not stop_found:
                if stop_key in line:
                    stop_found = True
                else:
                    lines.append(line)
            elif start_key in line and not stop_key in line:
                lines.append(line)
                start_found = True
                stop_found = False
    return lines


def findings(file_path):
    start_key = "FINDINGS:"
    stop_key = "IMPRESSION:"
    start_found = False
    stop_found = False
    lines = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if start_found and not stop_found:
                if stop_key in line:
                    stop_found = True
                else:
                    lines.append(line)
            else:
                if start_key in line:
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
    report_files = [f for f in listdir(report_path) if isfile(join(report_path, f)) and f.endswith('.txt')]
    output_path = './data/input/SDS_PV2_combined/reports_single_find_impr'
    for report in report_files:
        out_file = join(output_path,"{0}_fi.txt".format(report[0:-4]))
        rf = join(report_path,report)
        with open(out_file,'a') as f:
            f.write("{0} \n\n {1}".format(findings,impression))
            f.write(string.join(findings(rf)))
            f.write("\n\n")
            f.write(string.join(impression(rf)))

