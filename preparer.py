#!/usr/bin/env python

import numpy as np

# prepares command lines for the solver execution
# saves the command lines to a file

job_output_filename = 'solver_jobs.txt'

C_mole_percent_range = np.linspace(0.244792032777, 0.311870354205, 53)
H_mole_percent = 0

heating_rate_range = np.linspace(5, 120000, 53) / 60.
max_temperature_range = np.linspace(673, 1223, 53)

jobs = 0
with open(job_output_filename, "w") as o:
    for C_mole_percent in C_mole_percent_range:
        # compute PLIGC
        PLIGC = C_mole_percent 
        # compute PLIGO
        PLIGO = 1-C_mole_percent
        PLIGH = 0
        for heating_rate in heating_rate_range:
                for max_temperature in max_temperature_range:
                    o.write("./solver.py -C %f -H %f -O %f -r %f -m %f\n" %
                        (PLIGC, PLIGH, PLIGO, heating_rate, max_temperature))
                    jobs += 1

print("wrote %d job command lines to file '%s'" % (jobs, job_output_filename))
print("run the jobs with this command:\nparallel -a %s --max-procs 16" % job_output_filename)
