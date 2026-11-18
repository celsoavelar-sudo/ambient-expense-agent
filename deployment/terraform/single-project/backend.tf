terraform {
  backend "gcs" {
    bucket = "ctsal200-terraform-state"
    prefix = "ambient-expense-agent/dev"
  }
}
