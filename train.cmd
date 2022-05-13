executable = train.sh
getenv     = true
input      = train.cmd
output     = train.output
error      = train.error
log        = train.log
notification = complete
transfer_executable = false
request_memory = 2*1024
request_GPUs = 1
Requirements = (Machine != "patas-gn1.ling.washington.edu")
queue