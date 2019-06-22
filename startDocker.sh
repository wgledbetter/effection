#!/bin/bash

sudo service docker start
# docker build . -t wgledbetter/effection
docker run -it -v $PWD:/home/ wgledbetter/effection /bin/bash