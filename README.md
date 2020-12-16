# Instructions on Publising 
### Step 1.
Installing bower components
```
./setup.sh
```
### Step 2.
Using `gulp` to compile the web content from our source code. The configuration that `gulp` command will follow is *gulpfile.js* by default.
```
gulp
```
### Step 3.
The updated website content will be generated under `GrailQA/`. Push it to branch `leaderboard`.


# Instructions on Updating Leaderboard
This website reads in the leaderboard statistics from *output.json*, so to update the leadboard, you only need to update *output.json* and re-publish it following the above instructions. To update *output.json* that contains the latest information from codalab server, following the instructions under `grailqa/`. Currently, only [Yu Gu](gu.826@osu.edu) is able to do this.
