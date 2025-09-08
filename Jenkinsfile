pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'
        AWS_ACCOUNT_ID = '047719629738'
        REPO_NAME = 'my-repo'
        ECR_REPO = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"
        IMAGE_TAG = "build-${BUILD_NUMBER}"
    }

    stages {
        stage('Clone GitHub Repo') {
            options { timeout(time: 5, unit: 'MINUTES') }
            steps {
                script {
                    echo 'Cloning GitHub repo to Jenkins...'
                    checkout scmGit(
                        branches: [[name: '*/main']],
                        extensions: [],
                        userRemoteConfigs: [[
                            credentialsId: 'github-token',
                            url: 'https://github.com/priyankas247/RAG-MEDICAL-CHATBOT.git'
                        ]]
                    )
                }
            }
        }

        stage('Login to AWS ECR') {
            options { timeout(time: 5, unit: 'MINUTES') }
            steps {
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
                    sh """
                        aws ecr get-login-password --region ${AWS_REGION} \
                        | docker login --username AWS --password-stdin ${ECR_REPO}
                    """
                }
            }
        }

        stage('Build Docker Image') {
            options { timeout(time: 20, unit: 'MINUTES') }
            steps {
                sh """
                    docker build -t ${ECR_REPO}:${IMAGE_TAG} \
                                 -t ${ECR_REPO}:latest .
                """
            }
        }

        stage('Push to ECR') {
            options { timeout(time: 30, unit: 'MINUTES') }
            steps {
                sh """
                    docker push ${ECR_REPO}:${IMAGE_TAG}
                    docker push ${ECR_REPO}:latest
                """
            }
        }
    }

    post {
        always {
            echo 'Cleaning up Docker...'
            sh 'docker system prune -af --volumes || true'
        }
    }
}








        //  stage('Deploy to AWS App Runner') {
        //     steps {
        //         withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
        //             script {
        //                 def accountId = sh(script: "aws sts get-caller-identity --query Account --output text", returnStdout: true).trim()
        //                 def ecrUrl = "${accountId}.dkr.ecr.${env.AWS_REGION}.amazonaws.com/${env.ECR_REPO}"
        //                 def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"

        //                 echo "Triggering deployment to AWS App Runner..."

        //                 sh """
        //                 SERVICE_ARN=\$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text --region ${AWS_REGION})
        //                 echo "Found App Runner Service ARN: \$SERVICE_ARN"

        //                 aws apprunner start-deployment --service-arn \$SERVICE_ARN --region ${AWS_REGION}
        //                 """
        //             }
        //         }
        //     }
        // }
    
