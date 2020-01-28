#!/usr/bin/env python3

#-------------------------------------------------------------------------
# cancel
#
# This file simplifies the process of canceling jobs on the cluster.
# It parses input arguments that describe the job/tasks that should be
# canceled, and calls qdel with the appropriate arguments.
#-------------------------------------------------------------------------


import argparse
import datetime
import os
import re
import subprocess
import sys
import time

def parse_args():
    # Parse input arguments
    #   Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j','--jobid', help='The job ID to delete', type=str, default=None, required=True)
    parser.add_argument('-t','--tasklist', help='Comma separated list of task IDs to submit (e.g. "18-22:1,26,29,34-49:1")', type=str, default=None)
    parser.add_argument('-y','--dry_run', help="Don't actually submit jobs to grid engine", action='store_true')
    parser.set_defaults(dry_run=False)
    return parser.parse_args()
args = parse_args()


cmd = "qdel {} ".format(args.jobid)
if args.tasklist is not None:
    cmd += "-t {taskblock}"

def cancel(cmd):
    print(cmd)
    if not args.dry_run:
        try:
            subprocess.call(cmd, shell=True)
        except (subprocess.CalledProcessError, ValueError) as e:
            print(e)
            sys.exit()

if args.tasklist is None:
    yn = input('Are you sure you want to cancel all tasks for this job? (y/N)\n> ')
    if yn in ['y','yes','Y',"YES"]:
        cancel(cmd)
    elif yn in ['n','no','N',"NO",'']:
        print('Job cancellation aborted.')
else:
    taskblocks = args.tasklist.split(',')
    for taskblock in taskblocks:
        cancel(cmd.format(taskblock=taskblock))
