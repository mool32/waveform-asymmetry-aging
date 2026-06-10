INSTRUCTIONS TO REPRODUCE RESULTS FROM PAPER

+ install CaImAn from repository https://github.com/flatironinstitute/CaImAn/ 
+ use the commit with tag "paper_results_resubmission" from github for the exact reproduction of results, 
+ otherwise just use the latest version (suggested, results might vary slightly)

ALL THE SCRIPTS FILES MUST BE RUN USING THE ROOT DROPBOX FOLDER AS BASE FOLDER (this means you might have to set some variables to point to the right files) 

!! THIS STEP CAN BE SKIPPED IS LOADING DIRECTLY THE PREPROCESSED RESULTS !!
!! THIS STEP CAN TAKE A LOT OF RAM MEMORY AND TIME SINCE OPTIMIZED FOR WORKSTATIONS !!
+ generate results file for each dataset (this step can be skipped is loading directly the preprocessed results): 
	1. edit the Preprocess_batch.py and Preprocess_CaImAn_online.py file to point to the right folder 
	2. run the scripts setting the appropriate flags to save the data
        reload = False
        plot_on = False
        save_on = False  # set to true to recreate results for each file
        save_all = True
	3. check that memory is within boundary, otherwise reduce  values of the variable "n_processes"

+  Print figures
	+ Simply RUN the figure_xxx  scripts AS THEY ARE pointing to the right base_folder



 

