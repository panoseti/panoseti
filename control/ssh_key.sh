#! /bin/bash

if [ $# != 1 ] ; then 
echo "Pease specify the user and remote_host."
echo "You can use the script like this: ./ssh_key.sh wei@192.168.1.100"
exit 1;
fi

ssh-keygen -t ecdsa -b 521
ssh-copy-id -i ~/.ssh/id_ecdsa.pub $1