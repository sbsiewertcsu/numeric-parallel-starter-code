# Instructions for [GlobalProtect-openconnect](https://github.com/yuezk/GlobalProtect-openconnect) CSU, Chico Linux Users

## 1. Ensure you do **NOT** have globalprotect installed.

You want to see this:

```{sh}
$ which globalprotect
globalprotect not found
```

If `globalprotect` is installed, be sure to remove it before continuing or it will cause errors.

## 2. Install GlobalProtect-openconnect

Instructions [here](https://github.com/yuezk/GlobalProtect-openconnect), but I'll show the command for Debian/Ubuntu below:

```{sh}
sudo add-apt-repository ppa:yuezk/globalprotect-openconnect
sudo apt-get install globalprotect-openconnect
```

## 3. (optional) Verify Installation with GUI

The GUI version is installed alongside the CLI version. 
GUI is free for two weeks, while CLI is open-source and always free.
If GUI works, CLI should too.

```{sh}
gpclient launch-gui
```

Now the window will pop up. Enter `vpn.csuchico.edu`, then log in and authenticate through DUO.

Be sure to close the GUI vpn before attempting to connect via CLI. (check htop and search for gpclient if unsure)

## 4. Functions to Connect and Disconnect via CLI

I would recommend adding these to your `~/.bash_aliases` file or equivalent.

```{sh}
# Connect to the VPN. 
# `sudo -Eb` lets gpclient keep permissions and runs it in the background.
# `-qq` suppresses output. 
vpn_connect() {
    sudo -Eb gpclient --fix-openssl connect -qq vpn.csuchico.edu
}

# Closes the VPN with SIGINT (same as ^C if it were running in foreground)
vpn_disconnect() {
    sudo pkill -SIGINT gpclient
}
```

## 5. Verify Connection

Simplest way is to try and ssh into ecc. 

```{sh}
ssh <user>@ecc-linux.csuchico.edu
```

`<user>` should be the same prefix as your .csuchico.edu email. 
For example, `ssh jbcollins@ecc-linux.csuchico.edu`

If it hangs, you're not connected.

To check for errors, run `sudo -E gpclient --fix-openssl connect vpn.csuchico.edu` (running in foreground without log suppression)

You may need to instead run `sudo -E gpclient --fix-openssl connect --clean vpn.csuchico.edu` (clears cookies), or `sudo -E gpclient --fix-openssl connect --hip vpn.csuchico.edu` (accomodates hip, but stopped working for me), though these are not guaranteed to work.

The version of GlobalProtect used by CSU Chico is outdated and no longer supported by the packages listed on the support page, hence the need for this project.
