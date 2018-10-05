#!/bin/bash

# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OS="$(uname)"

if [ $OS == "Linux" ]
then

	apt-get update
        if [[ ($# -gt 0) && ($1 == "--oracle") ]]
          then
            add-apt-repository ppa:webupd8team/java
            apt-get update
            apt-get install oracle-java8-installer
          else
            apt-get install openjdk-8-jdk
        fi
        apt-get install maven python3
				apt-get install python3-flask
				apt-get install python3-pip
				pip3 install -U nltk
				pip3 install -U scikit-learn
				pip3 install --upgrade tensorflow
				pip3 install -U matplotlib


elif [ $OS == "Darwin" ] # Darwin for Mac OS X
then
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	brew update
        brew tap caskroom/versions
	brew cask install java8
	brew install maven
        brew install python3

else
	echo "Unknown OS."
	exit
fi
