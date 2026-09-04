# Security

This document details the security configuration required for the solution. In particular, it covers:

- **HTTPS Setup**
- **Image URL Fetching (SSRF)**

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

## 2. Image URL Fetching (SSRF)

### Overview

The multimodal API accepts an `image_url` content part that is either an inline base64 `data:` URL or a
remote URL. When a remote URL is given, the gateway makes the outbound request itself, so a caller who
can reach the API can influence where the gateway connects. Without validation, that is a server-side
request forgery (SSRF) vector: a request could point the gateway at endpoints only reachable from the
task or instance, and the fetched bytes would be sent to the model and returned in the completion.
On ECS/Fargate the notable targets are:

- the container credential endpoint, `169.254.170.2`, which serves the task role's temporary credentials
- the EC2 instance metadata service, `169.254.169.254`
- `localhost` and any VPC-internal service reachable from the task's subnet and security group

### What the gateway enforces

Remote image URLs are validated in [`src/api/image_url.py`](../src/api/image_url.py) before any request
is made. A URL is only fetched when all of the following hold:

| Check | Behaviour |
|---|---|
| Scheme | Only `http` and `https`. `file:`, `gopher:`, `ftp:`, `dict:` and others are rejected. |
| Address | Every address the host resolves to must be globally routable. This blocks the link-local range that serves both metadata endpoints, as well as loopback, private, shared address space, reserved and multicast ranges, including their IPv4-mapped IPv6 forms. |
| Redirects | Followed manually, up to 3 hops, and re-validated at every hop so a public URL cannot redirect to an internal one. |
| Response size | Capped, so a large or endless response cannot exhaust task memory. |
| Content type | Normalised to a format Bedrock accepts (`image/jpeg`, `image/png`, `image/gif`, `image/webp`). |

Rejected requests return `400` with a generic message, so error responses cannot be used to probe which
internal hosts exist.

### Recommended configuration

The checks above resolve the host and then let the HTTP client resolve it again when connecting. A
hostname served by an attacker-controlled DNS server with a very short TTL could in principle return a
permitted address to the check and an internal address to the connection (DNS rebinding). To rule that
out, and to reduce the blast radius generally, apply the controls that fit your deployment:

```bash
# Only allow images from hosts you trust. Strongest option; recommended if you know the hosts.
IMAGE_URL_ALLOWED_HOSTS=images.example.com,cdn.example.com

# Or refuse remote URLs entirely and require callers to send base64 data URLs.
ENABLE_IMAGE_URL_FETCH=false

# Lower the response size cap from the 10 MB default if your images are smaller.
IMAGE_URL_MAX_SIZE_MB=5
```

Set these as environment variables on the ECS task definition or the Lambda function.

In addition:

- **Scope the task role.** This is the control that limits the damage if the credential endpoint is ever
  reached. The bundled templates already grant the task role only `bedrock:InvokeModel`,
  `bedrock:InvokeModelWithResponseStream`, `bedrock:ListFoundationModels` and
  `bedrock:ListInferenceProfiles`. Keep it that narrow; credentials obtained through any future
  SSRF-style issue then grant nothing beyond model invocation.
- **Restrict egress** to limit which VPC-internal services and internet hosts the task can reach. Note
  the limit of this control: security groups do **not** filter link-local traffic, so an egress rule
  cannot protect the container credential endpoint at `169.254.170.2` or the EC2 instance metadata
  service at `169.254.169.254`. Those are served locally and never traverse the ENI where the security
  group applies.
- **Require IMDSv2** if you run the gateway on EC2 or ECS-on-EC2, so instance metadata cannot be read
  with a plain `GET`. On ECS-on-EC2 you can also block container access to instance metadata entirely
  with `ECS_AWSVPC_BLOCK_IMDS=true` in `/etc/ecs/ecs.config`. Neither setting applies to Fargate.

### If you deployed an earlier version

Deployments created before this hardening was added pass the caller-supplied image URL straight to the
HTTP client with no validation. If you are running the gateway from an image built before this change,
treat it as exposed to the SSRF vector described above.

**Redeploy** from the current version, or rebuild your container image from the current source. See
[How to upgrade?](../README.md#how-to-upgrade) in the README.

Redeploying is the mitigation, not merely the preferred one. Be aware that on Fargate there is no
network-level workaround: as noted above, security groups cannot filter the link-local credential
endpoint, and `ECS_AWSVPC_BLOCK_IMDS` is not available on Fargate. The environment variables in this
section only exist in the hardened code, so they cannot mitigate an older image either.

Until you can redeploy, the controls that do still apply are:

- Confirm the task role carries no permissions beyond Bedrock invocation, so any credentials reachable
  through the vector are of limited value.
- Treat the API key as the only thing standing between a caller and this vector: rotate it if it may have
  been shared beyond its intended users, and restrict who can reach the ALB or API Gateway endpoint.

Note that you cannot determine retrospectively whether the vector was probed. The gateway does not log
request bodies, so an older image leaves no record of the `image_url` values it fetched; absence of log
evidence is not evidence that nothing was attempted. After upgrading, rejected URLs are recorded as
warnings in CloudWatch, so probing becomes visible from that point on.

---

By following the steps outlined in this guide, you can configure a secure environment that uses HTTPS via ALB for encrypted traffic.