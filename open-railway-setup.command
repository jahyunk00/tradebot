#!/bin/bash
cd "$(dirname "$0")"
chmod +x deploy/railway/setup_railway.sh
exec deploy/railway/setup_railway.sh
