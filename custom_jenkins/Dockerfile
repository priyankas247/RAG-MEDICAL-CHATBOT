# Use the Jenkins image as the base image
FROM jenkins/jenkins:lts

# Switch to root to install system dependencies
USER root

# Install Docker and AWS CLI
RUN apt-get update && \
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        unzip \
        software-properties-common

# Install Docker (same as before)
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

# Add Jenkins user to the docker group
RUN groupadd -f docker && usermod -aG docker jenkins

# Create Docker volume
VOLUME /var/lib/docker

# Switch back to Jenkins user
USER jenkins
