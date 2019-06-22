#!/bin/bash

sudo service docker start
docker build . -t wgledbetter/effection
docker run -it -v ../effection/:/home/ wgledbetter/effection /bin/bash