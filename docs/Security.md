# Security

This document details the security configuration required for the solution. In particular, it covers:

- **HTTPS Setup**

Following these guidelines will help ensure that traffic is encrypted over the public network.

---

## 1. HTTPS Authentication with the ALB

### Overview

Using HTTPS on your ALB guarantees that all client-to-ALB communication is encrypted. This is achieved by:
- **Obtaining and managing SSL/TLS certificates** using AWS Certificate Manager (ACM). You'll need a domain but you can request a free certificate.
- **Configuring HTTPS listeners** on the ALB
- **Automating HTTP to HTTPS redirect** for clients that inadvertently access HTTP endpoints
- **Allowing traffic in the Security Group of the ALB**

### Step-by-Step Setup

#### 1.1. Request an SSL/TLS Certificate via ACM

1. **Navigate to AWS Certificate Manager (ACM):**  
   In the AWS Management Console, go to ACM in the region where your ALB is deployed.

2. **Request the Certificate:**  
   - Click on **"Request a certificate"**.
   - Choose **"Request a public certificate"** (or a private one if using a private CA).
   - Enter your domain names (e.g., `example.com`, `*.example.com`).
   - Complete the validation (via DNS or email). DNS validation is generally preferred for automation purposes.

3. **Certificate Validation:**  
   Ensure that the certificate status becomes **"Issued"** before proceeding.

#### 1.2. Configure the ALB for HTTPS

1. **Create or Modify the ALB Listener:**  
   - Open the **EC2 Dashboard** and navigate to [Load Balancers](https://console.aws.amazon.com/ec2/home?#LoadBalancers:).
   - If you already have an ALB, select it; otherwise, create a new ALB.
   - Under the **Listeners** tab, click **Manage listener** > **Edit Listener**.
   - Configure the listener protocol to **HTTPS** with port **443**.
   - Select the certificate you requested from ACM.

#### 1.3. (Optional) Redirect HTTP Traffic to HTTPS

To enhance security, ensure that any HTTP requests are automatically redirected to HTTPS.

1. **Create an HTTP Listener on Port 80:**
   - Add a listener on port **80**.
   - In the listener settings, add a rule to redirect all traffic to port **443** with the protocol changed to **HTTPS**.
     
   **Example AWS CLI command for redirection:**
   ```bash
   aws elbv2 create-listener \
       --load-balancer-arn <your-alb-arn> \
       --protocol HTTP \
       --port 80 \
       --default-actions Type=redirect,RedirectConfig="Protocol=https,Port=443,StatusCode=HTTP_301"
   ```

#### 1.4. Allow traffic in the Security Group of the ALB

1. **Create a Security Group:**
   - Go to the CloudFormation stack you originally used to deploy, select **Resources** and search for **ProxyALBSecurityGroup**
   - Click on the Security Group
   - Edit the Inbound Rules to allow traffic on Port 443 from `0.0.0.0/0` and (optionally) delete the Inbound Rule on Port 80. **Note**: If you delete the rule on port 80, you will need to update the base url to use HTTPS only as it won't redirect HTTP traffic to HTTPS.

Now you should be able to test your application! Use the base url like:

```
https://<your-domain>/api/v1
```

---

By following the steps outlined in this guide, you can configure a secure environment that uses HTTPS via ALB for encrypted traffic.