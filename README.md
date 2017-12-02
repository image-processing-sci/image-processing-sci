# image-processing-sci
GT Smart City Infrastructure's Image Processing Team (Docker)

This repository contains our code for processing the videos provided by the Smart City Infrastructure team.

Note: This repository doesn't contain any proprietary information/videos. For those, please email Cibi, Dr. Tsai, or Anirbhan

# Getting Started

1) Clone this repository (note if you're using ssh you should set up your keys by following the directions [here] (https://help.github.com/articles/connecting-to-github-with-ssh/))

    `git clone git@github.com:image-processing-sci/image-processing-sci.git` or `git clone https://github.com/image-processing-sci/image-processing-sci.git`

2) Open up the repository in terminal.

3) Create folder called `big_files`.

4) Place a video called `final.mp4` in `big_files` and the `background.png` image.

5) In the root of the folder, use the following command to get up and running

    `python3 background_subtractor.py`

Voila, you should see some windows of the processed videos.

# Contributing

We follow the standard industry process of contribution. If you make a change to a file, and want to merge it, you will need to open up a pull request.
Here's how to do it:

1) Make a change to a file

2) `git add .`

3) `git commit -m "<your commit message>"`

4) `git push origin master:YOUR_BRANCH_NAME`

5) Navigate back to the github page, link [here] (https://github.com/image-processing-sci/image-processing-sci) for convenience.

6) You should see an option to open up a pull request. Use that button and write in some details about what you changed.

7) After submitting the pull request, we run a few tests to ensure that what is being merged onto master is good. Hence we have required a Travis integration test and code cov test. If you want to see progress of this, or when a teammate opens up a PR, join the #ip_build channel on slack.

8) We also require one reviewer who is not the committer to approve the changes. Hence please ask a teammate to review, potentially offer feedback, and then approve your code. After this you will be able to merge.

# TODOs

1) Identify roadway assets, i.e signboards, cones.

2) Detect roadway pavement markings from a moving camera.

3) Categorize vehicles by type.
