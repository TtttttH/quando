# Quando - Visual Programming for Digital Interactive Exhibits

## To Deploy in Windows - tested with Windows 10 Pro

Prerequisites: Chrome browser, Node JS **Do not use Current - Use LTS version** (tested with v10.15.0), (optional) git for windows install (https://gitforwindows.org/)

1. Download the zip and extract into C:\quando, or Clone the repository in C: using git clone https://github.com/andrewfstratton/quando.git
2. In the command line, in C:\quando, npm update - this will likely take a while

### Setting up Quando for first time for (local) development

1. Run 'npm run install_local' to install pouchdb - you should only need to do this once
2. If you see errors for building sqlite3, try (for windows):
  * In an admin shell, run 'npm install -g windows-build-tools'
3. Run quando using 'npm run dev' (which will try to run the local pouchdb server and the quando node server).  Note that the PouchDB log file is in pouchdb/log.txt
4. Open the Control Panel through http://127.0.0.1 in chrome
5. Create a new user, e.g. 'test' with password 'test' and 'Add User'.
    * Note: you can change a user's password (or delete a user) through the PouchDB Control Panel - available through the hub page, or at http://127.0.0.1:5984/_utils.
6. ~~Chrome needs to be modified to allow video and audio to auto play:~~
    1. ~~_open chrome://flags/#autoplay-policy_~~ This has been removed - so Chrome will no longer play video/audio
    2. ~~_change to 'no user gesture is required'_~~
7. From the Control Panel, open 127.0.0.1/inventor using the QR Code or click the link for local access, login as test/test

### To add automatic Windows *Server* startup - for deployed use - not for development
1. using Windows R, run gpedit.msc
2. Choose Computer Configuration->Windows Settings->Scripts->Startup
    1. Then 'Add' C:\quando\quando.bat
    2. (optional) follow the next instructions for Client browser setup - *(where you have a client display running on the server as well)*
### Client browser Kiosk setup
_i.e. allow PC to boot into client browser interaction_
The following setup can be done (by itself) on any client machine - though kiosk.bat will need to be copied over

1. (if necessary) Edit the kiosk.bat file to change the location of Chrome
2. Then 
  * Either change the location of the chrome user data folder
  * Or create a folder c:\quando\chrome_user
  * Or remove the --user-data-dir xxx from the file
3. Save and Run quando\kiosk.bat
4. Then choose the interaction you want to automatically load on booting.
5. You can right click the screen to go back to the client setup.
6. using Windows R, run gpedit.msc
    * Choose Computer Configuration->Windows Settings->Scripts->Startup
    * Then 'Add' C:\quando\kiosk.bat to autostart Chrome

If everything is fine - then try restarting to see if everything boots correctly.

### Optional - Leap Motion
The standard Leap Motion (Orion) software needs to be installed on the Client PC, i.e. where the Leap motion is plugged in and where the browser will be run. The SDK is not needed.

Note: Web Apps must be enabled for using the leap Motion - in Windows, you may need to see https://forums.leapmotion.com/t/allow-web-apps-setting-resets-on-pc-on-computer-restart/8057

### Updating using Git
To update (assuming quando has changed), first kill, or exit, the Node.js process in the task manager,
then use:

* once only:
  * git update-index --assume-unchanged .\pouchdb\log.txt
  * git update-index --assume-unchanged .\pouchdb-config.json
* git pull origin master
* quando or npm start

## Editing as a Developer

The instructions below assume that you are using Visual Studio Code, though specifics are generally avoided.

Run the editor, then:
1. Run 'node .' or 'npm start', e.g. from the terminal
3. Open a Browser to 127.0.0.1

N.B. The client screen can be right clicked to allow you to select already deployed/created scripts - whichever one you open will be reopened next time you open 127.0.0.1/client.  This can also be done from the kiosk boot, so that a different interaction is loaded next time the PC reboots.

### Optional - auto reload
npm install -g nodemon
Then use nodemon instead of node

### Block Development

The (in progress) [Manifesto](docs/manifesto.md) is likely to be useful.

[Block Development](docs/creating_new_blocks.md)

### Installing in IBM Cloud

Quando can be deployed to IBM Cloud - note that this has not been tested.

[![Deploy to IBM Cloud](https://cloud.ibm.com/devops/setup/deploy/button.png)](https://cloud.ibm.com/devops/setup/deploy?repository=https%3A%2F%2Fgithub.com%2Fandrewfstratton%2Fquando.git&branch=master)
