pipeline {
    agent any

    // environment {
    //     AWS_REGION = 'us-west-2'
    //     ECR_REPO = 'my-repo'
    //     IMAGE_TAG = 'latest'
    //     SERVICE_NAME = 'llmops-medical-service'
    // }

    stages {
        stage('Clone GitHub Repo') {
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

//         stage('Build Docker Image') {
//             steps {
//                 script {
//                     echo "Building Docker image..."
//                     sh """
//                         docker build -t ${ECR_REPO}:${IMAGE_TAG} .
//                     """
//                 }
//             }
//         }

//         stage('Trivy Scan') {
//     steps {
//         script {
//             echo "Running Trivy scan..."
//             sh """
//                 docker run --rm \
//                     -v /var/run/docker.sock:/var/run/docker.sock \
//                     -v "${WORKSPACE}:/root" \
//                     aquasec/trivy image \
//                     --severity HIGH,CRITICAL \
//                     --format json \
//                     -o /root/trivy-report.json \
//                     "${ECR_REPO}:${IMAGE_TAG}" || true
//             """
//         }
//     }
// }



//         stage('Archive Trivy Report') {
//             steps {
//                 script {
//                     echo "Archiving Trivy report..."
//                     archiveArtifacts artifacts: 'trivy-report.json', allowEmptyArchive: true
//                 }
//             }
//         }

//         stage('Push to ECR') {
//             steps {
//                 withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
//                     script {
//                         def accountId = sh(script: "aws sts get-caller-identity --query Account --output text", returnStdout: true).trim()
//                         def ecrUrl = "${accountId}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
//                         def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"

//                         sh """
//                             aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ecrUrl}
//                             docker tag ${ECR_REPO}:${IMAGE_TAG} ${imageFullTag}
//                             docker push ${imageFullTag}
//                         """
//                     }
//                 }
//             }
//         }

        // Optional Deployment Stage
        // stage('Deploy to AWS App Runner') {
        //     steps {
        //         withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
        //             script {
        //                 def accountId = sh(script: "aws sts get-caller-identity --query Account --output text", returnStdout: true).trim()
        //                 def ecrUrl = "${accountId}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
        //                 def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"
        //                 
        //                 echo "Triggering deployment to AWS App Runner..."
        //                 
        //                 sh """
        //                     SERVICE_ARN=\$(aws apprunner list-services --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text --region ${AWS_REGION})
        //                     echo "Found App Runner Service ARN: \$SERVICE_ARN"
        //                     aws apprunner start-deployment --service-arn \$SERVICE_ARN --region ${AWS_REGION}
        //                 """
        //             }
        //         }
        //     }
        // }
    }
}
