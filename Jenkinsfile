pipeline {
  agent {
    kubernetes {
      inheritFrom 'inbound-agent'
      defaultContainer 'jnlp'
      label 'inbound-agent'
    }
  }

  environment {
    // Generate dynamic build version based on current date
    BUILD_VERSION = "${new Date().format('yy')}.${(new Date().format('MM') as Integer <= 3) ? 1 : (new Date().format('MM') as Integer <= 6) ? 2 : (new Date().format('MM') as Integer <= 9) ? 3 : 4}-SNAPSHOT-${BUILD_NUMBER}"
  }

  stages {
    stage('Build & Push Docker Image') {
      steps {
        script {
          echo "Building and pushing Docker image with version: ${env.BUILD_VERSION}"
          
          sh '''#!/bin/bash
            # Install required tools if not available
            if ! command -v docker &> /dev/null; then
              apt-get update
              apt-get install -y docker.io
            fi
            
            if ! command -v aws &> /dev/null; then
              apt-get update
              apt-get install -y awscli jq
            fi
            
            if ! command -v jq &> /dev/null; then
              apt-get install -y jq
            fi
            
            # Build the Docker image
            echo "Building Docker image: bedrock-access-gateway:${BUILD_VERSION}"
            docker build -t bedrock-access-gateway .
            
            # Tag image for ECR
            docker tag bedrock-access-gateway:latest 382254873799.dkr.ecr.us-east-1.amazonaws.com/bedrock-access-gateway:${BUILD_VERSION}
            
            # Set AWS region
            export AWS_DEFAULT_REGION=us-east-1
            
            # Assume the staging role for ECR access
            echo "Assuming AWS role for ECR push..."
            TEMP_ROLE=$(aws sts assume-role \\
              --role-arn "arn:aws:iam::382254873799:role/jenkins_ai-foundation_runway" \\
              --role-session-name "jenkins-docker-push-${BUILD_NUMBER}")
            
            export AWS_ACCESS_KEY_ID=$(echo $TEMP_ROLE | jq -r .Credentials.AccessKeyId)
            export AWS_SECRET_ACCESS_KEY=$(echo $TEMP_ROLE | jq -r .Credentials.SecretAccessKey)
            export AWS_SESSION_TOKEN=$(echo $TEMP_ROLE | jq -r .Credentials.SessionToken)
            
            # Login to ECR
            echo "Logging into ECR..."
            aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 382254873799.dkr.ecr.us-east-1.amazonaws.com
            
            # Push to ECR
            echo "Pushing image to ECR..."
            docker push 382254873799.dkr.ecr.us-east-1.amazonaws.com/bedrock-access-gateway:${BUILD_VERSION}
            
            echo "Successfully pushed image:"
            echo "382254873799.dkr.ecr.us-east-1.amazonaws.com/bedrock-access-gateway:${BUILD_VERSION}"
          '''
        }
      }
    }
  }
}