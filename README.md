# SURPRISE
## SURPRISE HOUSING PREDICTION

### Software requirements and tools



1.[Github Account](https://github.com)

2.[HerokuAccount](https://heroku.com)

3.[VSCODEIDE](https://code.visualstudio.com/)

4.[GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)


````````````````````````````````````````````````````````````````````````````````

Cloning Github repository to selected folder
-----------------------------------------------
git clone https://github.com/SrikanthAmb/SURPRISE.git

Place all the files like csv file, ipynb file, app.py files in folder "SURPRISE" file created by github clone command.
Create the folder "templates", place html page in it.
Create the folder and subfolder with names 'static' and 'images' and place image file in 'images' folder.

Create a new Environment
`````````````````````````
conda create -p surprisehouse python==3.10 -y

Activate the new Environment
------------------------------
conda activate file/path/to/surprisehouse

Connecting to Github Account
-----------------------------
git config --global user.name "SrikanthAmb"

git config --global user.email "ambidisrikanth@gmail.com"

Create Requirements list in .txt format
-----------------------------------------
list out all the necessary and used packages in jupyter notebook

Install the packages listed in requirements.txt
-------------------------------------------------
pip install -r file/path/to/requirements.txt

Create Proc file
------------------
Create Procfile in the same directory and write the below function

web: gunicorn app:app

Test the app file
-------------------
Type "python app.py" in cmd prompt, with which a https link is triggered and displayed. Copying and pasting 
the link in the browser's search bar, web page is displayed. If no errors found on submitting values and displaying desired ouput, then we can proceed to Heroku for deployment.

Docker file and main.yaml
----------------------------
Create Dockerfile in the same directory.
Create folder with name ".github" and subfolder "workflows". Create main.yaml file inside "workflows"

Heroku to Github
----------------
Record the HEROKU_API_KEY, HEROKU_APP_NAME, HEROKU_EMAIL from Heroku.
Place the above values in secret repositories created in Github Action secrets.

Deployment
-------------
Connect to github repository "SURPRISE", and push the "Deploy" button.
You can view / open  the app on the up right corner.

`````````````````````````````````````````````````````````````````````````````````````````````````````````````

