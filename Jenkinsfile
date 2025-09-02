stage('Build, Scan, and Push Docker Image to ECR') {
    steps {
        withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-token']]) {
            script {
                def accountId = sh(script: "aws sts get-caller-identity --query Account --output text", returnStdout: true).trim()
                def ecrUrl = "${accountId}.dkr.ecr.${env.AWS_REGION}.amazonaws.com/${env.ECR_REPO}"
                def imageFullTag = "${ecrUrl}:${IMAGE_TAG}"

                sh """
                echo "🔐 Logging into AWS ECR..."
                aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ecrUrl}

                echo "🐳 Building Docker image..."
                docker build -t ${env.ECR_REPO}:${IMAGE_TAG} .

                echo "🔍 Running Trivy scan..."
                docker run --rm \
                  -v /var/run/docker.sock:/var/run/docker.sock \
                  -v ${env.WORKSPACE}:${env.WORKSPACE} \
                  -w ${env.WORKSPACE} \
                  aquasec/trivy image \
                  --scanners vuln \
                  --severity HIGH,CRITICAL \
                  --format json \
                  -o trivy-report.json \
                  ${env.ECR_REPO}:${IMAGE_TAG} || true

                echo "📦 Pushing Docker image..."
                docker tag ${env.ECR_REPO}:${IMAGE_TAG} ${imageFullTag}
                docker push ${imageFullTag}
                """

                // Archive the scan report
                archiveArtifacts artifacts: 'trivy-report.json', allowEmptyArchive: true
            }
        }
    }
}
