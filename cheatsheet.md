#### 0. Logging In/Out
- `ssh <login>@athena.cyfronet.pl` - logs in to your Athena acc in current terminal session
- `exit` - logs out of Athena
- [ssh key-based authentication](https://guide.plgrid.pl/en/computing/ssh_key/) - (We decided not to use it on Athena)
- [Athena documentation](https://docs.cyfronet.pl/display/~plgpawlik/Athena#Athena-AccesstoAthena)

---

#### 1. Linux File System
1. [ ] `.` - current directory
2. [ ] `..` - one directory up
3. [ ] [NOTE] - `<filename>` always refers to a filename in **current** directory
- `pwd` - print name of current/working directory
- `cd <path>` - go to specified path
- `ls` - print current directory's contents (`ls -l` for detailed info)
- `rm <filename>` - removes specified file (careful - no auth, I recommend `rm -i` for confirmation prompt)
- `rm -rf <path>` - removes recursively + force remove **everything** we have access to in specified directory. **Extreme caution advised**
*(despite filename also being proper argument in rm -rf, it should never be used with -r options. Because why would we :) )*
- `mkdir <dirname>` - creates new directory in current dir
- `rmdir <dirname>` - removes empty directory. Recommended when we are not deleting files

---

#### 2. Handling files 
- `cp <fileToBeCopied> <destinationDirectory>` - copies file to specified directory (-R for recursive copying - whole directories)
- `scp` - copying through SSH. Secure to transfer files **from your PC to Athena** (args and flags same as `cp` command)
- `scp -C` - copy **compressed** data through ssh.
    1. [ ] example: `scp -r ./MY_DIR <login>@athena.cyfronet.pl:/$HOME` will copy `MY_DIR` with all its contents to Athena `$HOME` directory
- `mv <sourceFilename> <destDir/destFilename>` - moves file to another directory with (optionally another name) moving to the same dir with different name changes the name. [Most of options with examples](https://www.geeksforgeeks.org/mv-command-linux-examples/)
- `cat <filename>` - prints file to terminal
- `vim <filename>` - opens file in the best text editor ever created
- TODO tars
***enviroment paths:***
    - `$HOME` - `/net/people/plgrid/<login>`
        - __10GB__ storage our configuration files & *own applications (?)*
    - `$SCRATCH` - `/net/tscratch/people/<login>`
        - High-speed storage. Data older than 30 days might be deleted without notice. My guess is to use it for computations, but store files and results in `$HOME/$PLG_GROUPS_STORAGE/<group name>` dir
    - `$PLG_GROUPS_STORAGE/<group name>` - `/net/pr2/projects/plgrid/<group name>`
        - __1TB__ Long-term storage for data needed for computations. Uses transparent compression (What does **transparent** mean here ?)

---

#### 3. Virtual Environment and modules
1. [ ] Located in `$SCRATCH/venv/bin`
-  `source $SCRATCH/venv/bin/activate` - activate venv
- `deactivate` - deactivate venv
- `module load  <name>/<version>` - loads spcified module
    - eg `module load GCC/12.3.0  OpenMPI/4.1.5 mpi4py/3.1.4`
    - *[modules available on Athena](https://guide.plgrid.pl/en/applications/athena/athena_apps/)*
    - `module list` - lists loaded modules `ml`
    - `module help` - help :pray:

---

#### 4. Computations
- Athena is using [**SLURM** resource manager](https://kdm.cyfronet.pl/portal/Podstawy:SLURM)
- We should submit jobs to `plgrid-gpu-a100`. *"Running extensive workloads not using GPUs will result in account suspension."* So careful XD
- `sbatch <scriptname>` - submit a batch script to Slurm. It has to be well configured.
    - When used in run script, `#SBATCH --option`
    - [sbatch manual](https://slurm.schedmd.com/sbatch.html)
- `srun` - "runs interactive job or step in batch job" - (im not quite sure here)
- **every** job has its unique `job_ID`

<br>

<details><summary>SAMPLE GPU JOB SCRIPT</summary>
#!/bin/bash<br>
#SBATCH --job-name=job_name<br>
#SBATCH --time=01:00:00<br>
#SBATCH --account=grantname-gpu<br>
#SBATCH --partition=plgrid-gpu-v100<br>
#SBATCH --cpus-per-task=4<br>
#SBATCH --mem=40G<br>
#SBATCH --gres=gpu<br>
<br>
module load cuda<br>
#
srun ./myapp<br></details>

---

#### 5. Queue System
- `squeue` - list of currently queued/running jobs
- `scontrol show job [<job_ID>]` - details of a given job
- `sstat -a -j <job_ID>/<job_ID.batch>` - resource consumption for each step of a currently running job
- `scancel <job_ID>` - cancel a queued or running job
- `sacct` - resource consumption for jobs/steps which have already been completed
- `scontrol show partition [<partition_name>]` - details of a given partition
- `sinfo` - list of nodes
- `scontrol show node [<node_name>]` - details of a given node
- `<command> --help` - hope it helps :pray:

--- 

#### 6. Recources
- `hpc-grants` - shows available grants, resource allocations, time
- `hpc-fs` - shows available storage
- `hpc-jobs` - shows currently pending/running jobs
- `hpc-jobs-history` - shows information about past jobs

---
[fine pdf with explanations and examples](https://docs.cyfronet.pl/display/~plgnoga/MCB+Cryo-EM+entry+training?preview=%2F74816825%2F74817087%2F2020-11-18-MCB-hpc-intro.pdf)
