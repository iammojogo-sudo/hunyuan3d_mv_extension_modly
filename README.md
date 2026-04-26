

FOR WINDOWS CURRENTLY. I DO NOT HAVE OTHER OPERATING SYSTEMS:


MODLY INSTALL FROM LIGHTNINGPIXEL's GITHUB REPO>>>>

IF YOU DO NOT HAVE MODLY YET> FOR A WINDOWS INSTALL AT LEAST, HERE ARE THE INSTRUCTIONS VIA GITHUB SINCE THOSE ARE THE DIRECT FILES: 

 Here are the commands for a clean install from GitHub on Windows (PowerShell):
>>>Step 1 — Clone the repo
powershell[as admin] 
'''git clone https://github.com/lightningpixel/modly.git'''
>>>NOTE: I FOUND OUT IF YOU OPEN POWERSHELL NOT AS ADMIN, JUST NORMALLY, IT SHOULD AUTOMATICALLY PUT THE FOLDER IN THE USER FOLDER WHERE WE WANT IT. THEREFORE, STEP TWO WOULD BE NOT NEEDED :)

>>>Step 2 -  move it somewhere else: recommended: 
'''Move-Item "C:\Windows\system32\modly" "C:\Users\$env:USERNAME\modly"'''
'''cd "C:\Users\$env:USERNAME\modly\api"'''
[MAKE SURE TO REPLACE USERNAME with your folder name for your user]

Thats the user folder so youll have less issues with powershell

>>>Step 3 — Set up the Python API backend
'''python -m venv .venv'''

NOTE: if python error: Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases. just run this:
'''winget install Python.Python.3.11 --override "/quiet InstallAllUsers=1 PrependPath=1"'''
close/reopen powershell (REQUIRED)

check version : '''python -version'''
then run:
'''cd "C:\Users\$env:USERNAME\modly\api"''' [MAKE SURE TO REPLACE USERNAME with your folder name for your user]
'''python -m venv .venv''' 

>>>Step 4 - activate the environment
'''.venv\Scripts\activate'''
>>>
IF ERROR WITH SCRIPT POLICY, run:
'''Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser'''
Y for yes
then again:
'''.venv\Scripts\activate'''

>>>Step 5 - install requirements in that environment:
'''pip install -r requirements.txt'''

'''cd ..''' < goes back to modly folder directly

Step 6 — Install frontend dependencies & run
powershell[STILL AS ADMIN REMEMBER] 
'''npm install'''

That needs Node.js! If you dont have it, it WILL error. 
If error: 
'''winget install OpenJS.NodeJS'''
WHen complete, THEN I would do close and reopen powershell[ADMIN] since it installed the js.

'''cd C:\Users\modly\api'''
(It has to reactivate the environment folder because it was closed), so again:
''' .venv\Scripts\activate'''   [IF NOT ALREADY IN (.venv)]

>>>Step 6 -
'''npm install''' - (I WOULDNT KNOW WHY IT WOULDNT INSTALL AT THIS POINT SO DONT ASK)
OR you could even be in the modly folder directly and run: 
'''cmd /c launch.bat''' since it has that file in it. [RECOMMENDED]

It should start the backend and run :) and NO FASTAPI ISSUES! which is good because if you tried to do some of this more 'manually', you would likely run into that issue :P

*The above was to install Modly via github to Windows from what I know. Other installs, search it up. I do not have Mac, Linux, etc
##########################

NOTE: If you happen to recieve a "Something went wrong", it might be from development files on the github BUT if it's related to python, its possible to fix on your end luckily. 
IF its something about a "Bundled Python" because tehnically if a program uses python, it should ahve it with the files", Then we can find the path or create it. 
The requirements should have taken care of that during download but try this: 
go back into the powershell under the modly folder still (admin), and type:
'''node scripts/download-python-embed.js'''
Thats a file in the scripts folder that does the python stuff for us that they made :) Make sure to thank them because it includes the environment! 

Youll see the folder populate if you are in there. 

###REQUIREMENTS FOR EXTENSION###
1. Download Git: https://git-scm.com/download/win
If Git is installed but not in PATH:

2. Open System Properties → Environment Variables

Add this to PATH:
C:\Program Files\Git\cmd

3. Restart Modly & Download the extension via Github
You can then try to download the extension via github here: https://github.com/iammojogo-sudo/hunyuan3d_mv_extension_modly

5. any setup.py errors during extension install:
Its likely either you are using OneDrive and/or have not allowed user permissions on files/folders, and/or it errored before but created a folder already in extensions that you need to delete from a previous error. Make sure modly installs directly on the system and you pick a data location on the system. 
If its not, uninstall Modly, reinstall on the system. 

6. If it successfully installs extension, MAKE SURE to download the weights from the Modly extensions panel with the purple "Download" button. Wait until finished before you do anything else. If you want, you can restart Modly after the weights are downloaded. Otherwise if you do not have the weights, it will not run. 

7. If you installed Modly directly, there are files to be replaced. I will contact the developer to include these in a next update and patch them in if they work and are tested by their team successfully. If not, I will fix them until they are ready for the update. UNTIL THEN, you will ONLY get to use 1 photo, which it defaults to "front" (can probably use any photo). Once the new files are ready, you will be able to utilize four photos

8. If it fails saying it cant find a model when the components are all there, you might need https://www.systoolsgroup.com/migrator/download-vc-redistributable.html. 

NOTE: the standard can work ok on 6GB VRAM but takes a while. It is recommended to have at least 8GB though. The other models are the same I believe and Turbo is actually more efficient on memory that standard anyhow. Fast is in between. Also, careful not to change panels when downloading weights as it might say they are done but they might not be. Best to close and reopen after a few minutes. 

Questions? Contact me or the Modly team as I will be giving all of this to them for future updates. 

FUTURE UPDATE: Added options to the mv generation as well as possible texturing ability since mv initially did not come with it. 

###DEVELOPER NOTES###

See `MODLY_CORE_NOTES.md` for the Modly core change needed to preserve named
multi-image model inputs in workflows. Until Modly maps model inputs by
`targetHandle`, connected workflow images still collapse to one primary front
image. The extension can already consume optional side views when Modly passes
them as `left_image_path`, `back_image_path`, and `right_image_path` params.



###ONCE EXTENSIONS ARE INSTALLED - DO NOT FORGET TO DOWNLOAD ANY WEIGHTS BEFORE TRYING TO USE THE EXTENSIONS###
