FOR WINDOWS CURRENTLY. I DO NOT HAVE OTHER OPERATING SYSTEMS:

1. Download Git: https://git-scm.com/download/win
If Git is installed but not in PATH:

2. Open System Properties → Environment Variables

Add this to PATH:
C:\Program Files\Git\cmd

3. Restart Modly

4. any setup.py errors during extension install:
Its likely either you are using OneDrive and/or have not allowed user permissions on files/folders, and/or it errored before but created a folder already in extensions that you need to delete from a previous error. Make sure modly installs directly on the system and you pick a data location on the system. 
If its not, uninstall Modly, reinstall on the system. 

5. If it successfully installs extension, MAKE SURE to download the weights from the Modly extensions panel with the purple "Download" button. Wait until finished before you do anything else. If you want, you can restart Modly after the weights are downloaded. Otherwise if you do not have the weights, it will not run. 

6. If you installed Modly directly, there are files to be replaced. I will contact the developer to include these in a next update and patch them in if they work and are tested by their team successfully. If not, I will fix them until they are ready for the update. UNTIL THEN, you will ONLY get to use 1 photo, which it defaults to "front" (can probably use any photo). Once the new files are ready, you will be able to utilize four photos

7. If it fails saying it cant find a model when the components are all there, you might need https://www.systoolsgroup.com/migrator/download-vc-redistributable.html. 

NOTE: the standard can work ok on 6GB VRAM but takes a while. It is recommended to have at least 8GB though. The other models are the same I believe and Turbo is actually more efficient on memory that standard anyhow. Fast is in between. Also, careful not to change panels when downloading weights as it might say they are done but they might not be. Best to close and reopen after a few minutes. 

Questions? Contact me or the Modly team as I will be giving all of this to them for future updates. 

FUTURE UPDATE: Added options to the mv generation as well as possible texturing ability since mv initially did not come with it. 
