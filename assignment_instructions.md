# SCI Module 2: Assignment Instructions

The assignments for this course will be distributed and collected through GitHub Classroom.  They will be in the form of jupyter notebooks.  Before you go through the following, first ensure you have followed all instructions in the ["setup.md"](https://github.com/summercomputinginstitute/demo_notebooks/blob/main/setup.md) document and are set up for work on your laptop or Colab.

# How to access assignments
To access the assignment, click on the assignment link from Sakai.  The first time you try to access an assignment from Module 2, when you click on the link you will be directed to a screen where you need to select your name.  This maps your GitHub username to your actual name so we know who is who.  You only need to do this once.  After you have clicked to accept the assignment, you will now see a new repository in your GitHub which contains the assignment.  This is a private repository, meaning only you and I (as the instructor) have access to it.  Each student has their own private repository containing a version of the assignment for them to edit.

# Working on assignments
## Option 1: work locally on your computer
### Download the assignment
- Create a folder for the assignment on your laptop  
- At the command line, change directory into the folder you just created  
- Download ("clone") the assignment to your local directory using the command: `git clone <assignment-repo-url>`.  Replace <assignment-repo-url> with the url of your assigment repository from GitHub.

### Opening the assignment
- At the command line, launch jupyter notebooks with the command: `jupyter notebook`
- Jupyter notebooks will open in your web browser  
- Navigate to the assignment notebook file (.ipynb ending) and open it

### Submitting your assignment
The assignment will be collected from your assignment GitHub repository.  You must push your work back up to GitHub for it to be available for collection and grading. You can push your work to GitHub as often as you would like. When you push your updated version, it will replace the original skeleton code version that was provided to you.  **Be sure that you do not change the name of the notebook file, and keep the file format as a .ipynb file**.

To push an updated version from your laptop back up to your GitHub repo, use the command line to navigate to the assignment directory and then run the following commands:
```
git add --all
git commit -m "<insert brief message about this version>"
git push
```
You can push new versions up to GitHub as often as you would like.  Whichever version exists as the latest in GitHub at time of collection will be the one that is graded.

## Option 2: Work in Colab
### Opening the assignment
Alternatively, you can work on your assignment notebook in Google Colab. From Colab, select `File -> Open notebook`.  Click on the "GitHub" tab and enter your GitHub username in the search box.  Then check the box that says "Include private repos".  You will see a pop-up window and be asked to allow Colab to access your GitHub. In the pop-up click "Authorize googlecolab".  You will then be able to select your private assignment repo from the Repository list and open it the notebook (.ipynb file) in Colab. 

### submitting your assignment
The assignment will be collected from your assignment GitHub repository.  You must push your work back up to GitHub for it to be available for collection and grading. You can push your work to GitHub as often as you would like. When you push your updated version, it will replace the original skeleton code version that was provided to you.  **Be sure that you do not change the name of the notebook file, and keep the file format as a .ipynb file**.

To push your work back to your GitHub from Colab, go to `File -> Save a copy in GitHub`. Keep the file name the same as the original, and then click 'ok'.  Your notebook file will be pushed to your GitHub repo.  You can push new versions up to GitHub as often as you would like.  Whichever version exists as the latest in GitHub at time of collection will be the one that is graded.

# View your graded assignment
Once the assignment has been graded, a graded version of the notebook will be pushed to your GitHub assignment repo. You will be able to view your grades for each part by viewing the notebook