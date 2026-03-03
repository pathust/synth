-- MySQL init script for synth_prices database
-- Creates user with caching_sha2_password (MySQL 8.0 default)
-- Because mysql_native_password is disabled in MySQL 8.0.36+ arm64 Docker

CREATE DATABASE IF NOT EXISTS synth_prices;

CREATE USER IF NOT EXISTS 'synth'@'%'
    IDENTIFIED BY '';

GRANT ALL PRIVILEGES ON synth_prices.* TO 'synth'@'%';
FLUSH PRIVILEGES;
