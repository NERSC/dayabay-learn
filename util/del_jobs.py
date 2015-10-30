__author__ = 'racah'
import subprocess
import time

def del_all_jobs(tries):
    queue_data = subprocess.check_output(['qstat', '-u racah']).split('\n')
    for line in queue_data[6:]:
        job_name = line.split(' ')[0]
        for tryy in range(tries):
           subprocess.call(['qdel', job_name])
      

if __name__ == "__main__":
    del_all_jobs(2)


