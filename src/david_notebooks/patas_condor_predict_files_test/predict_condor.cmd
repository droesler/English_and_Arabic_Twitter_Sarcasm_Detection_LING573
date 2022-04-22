# Condor control file

universe	= vanilla
executable	= run_predict_sarcasm.sh
transfer_executable = false
getenv		= true
output		= output
error       = error
Log		    = test.log
arguments   = ""
request_memory	= 2*1024
Requirements = ( Machine != "patas-n3.ling.washington.edu" )
Queue

