import glob
import os

filename = 'domains/cube/buchner2018/problems-with-halfturns/id-000_turns-1_problem-0.sas'

def remove_half_turns(filename='domains/cube/buchner2018/problems-with-halfturns/id-000_turns-1_problem-0.sas'):
    with open(filename, 'r') as file:
        lines = file.readlines()

    output = []

    # Grab header up to first operator
    first_operator_pos = lines.index('begin_operator\n')
    output.extend(lines[:first_operator_pos])
    del lines[:first_operator_pos]

    output[-1] = output[-1].replace('18','12')# Modify the number of operators

    while lines[0].strip('\n') == 'begin_operator':
        current_operator_name = lines[1].strip('\n')
        current_operator_end = lines.index('end_operator\n')
        # print('Found operator:', current_operator_name)
        if '2_0' in current_operator_name:
            # print('Half-turn operator. Removing...')
            pass
        else:
            # print('Quarter-turn operator. Saving...')
            output.extend(lines[:current_operator_end+1])

        del lines[:current_operator_end+1]

    output.extend(lines)
    del lines

    new_dir = 'domains/cube/buchner2018/problems'
    new_filename = os.path.split(filename)[-1]
    new_filepath = os.path.join(new_dir,new_filename)
    with open(new_filepath, 'w') as file:
        file.writelines(output)
    print('Wrote modified domain file to:', new_filepath)

if __name__ == "__main__":
    for filename in glob.glob('domains/cube/buchner2018/problems-with-halfturns/id-*.sas'):
        remove_half_turns(filename)