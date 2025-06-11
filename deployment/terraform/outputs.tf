output "api_base_url" {
  description = "Proxy API Base URL (OPENAI_API_BASE)"
  value       = "${aws_api_gateway_deployment.api_deployment.invoke_url}${aws_api_gateway_stage.api_stage.stage_name}/api/v1"
}
