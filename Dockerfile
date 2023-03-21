 # Author  : Serena G. Lotreck <lotrecks@msu.edu>
  
 # Instantiate Ubuntu 20.04
 FROM ubuntu:20.04
 LABEL maintainer "Serena Lotreck <lotrecks@msu.edu>"
 LABEL description="Docker image for running OpenIE"
  
 # Update Ubuntu Software repository
 RUN apt update
 RUN apt -y install python3-pip
 RUN apt -y install vim

 # Create working directory with the contents of the volume
 RUN mkdir /obie-wan
 ADD . /obie-wan
 WORKDIR /obie-wan

 # Install requirements
 RUN pip install --upgrade pip

 # TZ config lines from https://dev.to/setevoy/docker-configure-tzdata-and-timezone-during-build-20bk
 ENV TZ=America/New_York
 RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
 
 # Java installation lines in next three blocks from https://stackoverflow.com/a/44058196
 # Install OpenJDK-8
 RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;
    
 # Fix certificate issues
 RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

 # Setup JAVA_HOME -- useful for docker commandline
 ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
 RUN export JAVA_HOME
 
 RUN pip install jsonlines
 RUN pip install pandas
 RUN pip install stanza
 RUN pip install stanford-openie
 RUN pip install transformers

 
 