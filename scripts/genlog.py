#!/usr/bin/env python3
"""
Apache Access Log Generator
============================
Generates a synthetic Apache "combined" format access log (.txt) with a
customizable total number of lines, number of error responses (4xx/5xx,
excluding 401/403), and number of unauthorized-access responses (401/403).

Usage examples:
    python genlog.py --total 1000 --errors 50 --unauthorized 20
    python genlog.py -t 5000 -e 200 -u 100 -o my_access.log.txt
    python genlog.py -t 2000 -e 100 -u 50 --seed 42

Run with -h / --help to see all options.
"""

import argparse
import random
from datetime import datetime, timedelta


# --------------------------------------------------------------------------- #
# Sample data pools used to build realistic-looking log lines
# --------------------------------------------------------------------------- #

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]

PATHS = [
    "/", "/index.html", "/about", "/contact", "/products", "/products/123",
    "/cart", "/checkout", "/login", "/logout", "/register", "/admin",
    "/admin/dashboard", "/admin/users", "/api/v1/users", "/api/v1/orders",
    "/api/v1/products/42", "/api/v1/auth/token", "/static/css/style.css",
    "/static/js/app.js", "/static/img/logo.png", "/blog", "/blog/post-1",
    "/blog/post-2", "/search", "/favicon.ico", "/robots.txt", "/wp-login.php",
    "/config.php", "/.env", "/uploads/file.pdf",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "curl/8.4.0",
    "python-requests/2.31.0",
]

REFERRERS = [
    "-",
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "https://example.com/",
    "https://www.facebook.com/",
]

# Status codes grouped by category
SUCCESS_CODES = [200, 200, 200, 201, 204, 301, 302, 304]
ERROR_CODES = [400, 404, 404, 405, 408, 500, 502, 503, 504]
UNAUTHORIZED_CODES = [401, 403]


def random_ip():
    """Return a random-looking public IPv4 address."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def format_apache_time(dt):
    """Format a datetime as Apache log timestamp, e.g. [21/Jun/2026:14:23:01 +0000]"""
    return dt.strftime("[%d/%b/%Y:%H:%M:%S +0000]")


def build_log_line(dt, category):
    """Build a single Apache combined-format log line for the given category."""
    ip = random_ip()
    ident = "-"
    user = "-" if category != "success" else random.choice(["-", "-", "-", "john", "alice"])
    method = random.choice(HTTP_METHODS)
    path = random.choice(PATHS)
    protocol = "HTTP/1.1"

    if category == "unauthorized":
        status = random.choice(UNAUTHORIZED_CODES)
    elif category == "error":
        status = random.choice(ERROR_CODES)
    else:
        status = random.choice(SUCCESS_CODES)

    size = random.randint(150, 50000) if status != 204 else 0
    referrer = random.choice(REFERRERS)
    user_agent = random.choice(USER_AGENTS)

    request = f"{method} {path} {protocol}"
    timestamp = format_apache_time(dt)

    return (
        f'{ip} {ident} {user} {timestamp} "{request}" {status} {size} '
        f'"{referrer}" "{user_agent}"'
    )


def generate_log_lines(total, errors, unauthorized, start_time=None):
    """
    Build a list of `total` log lines, with `errors` of them being generic
    error codes and `unauthorized` of them being 401/403, the rest being
    successful responses. Lines are returned in chronological order.
    """
    success_count = total - errors - unauthorized

    categories = (
        ["unauthorized"] * unauthorized
        + ["error"] * errors
        + ["success"] * success_count
    )
    random.shuffle(categories)

    if start_time is None:
        start_time = datetime.now() - timedelta(hours=1)

    lines = []
    current_time = start_time
    for category in categories:
        # Random gap between requests, between 0 and 5 seconds
        current_time += timedelta(seconds=random.uniform(0, 5))
        lines.append(build_log_line(current_time, category))

    return lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Apache access log (.txt) file."
    )
    parser.add_argument(
        "-t", "--total", type=int, default=1000,
        help="Total number of log lines to generate (default: 1000)",
    )
    parser.add_argument(
        "-e", "--errors", type=int, default=50,
        help="Number of error lines, e.g. 400/404/500/502/503 (default: 50)",
    )
    parser.add_argument(
        "-u", "--unauthorized", type=int, default=20,
        help="Number of unauthorized-access lines, status 401/403 (default: 20)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="apache_access.txt",
        help="Output file path (default: apache_access.txt)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible output (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.total < 0 or args.errors < 0 or args.unauthorized < 0:
        raise SystemExit("Error: total, errors, and unauthorized must be >= 0.")

    if args.errors + args.unauthorized > args.total:
        raise SystemExit(
            f"Error: errors ({args.errors}) + unauthorized ({args.unauthorized}) "
            f"cannot exceed total ({args.total})."
        )

    lines = generate_log_lines(args.total, args.errors, args.unauthorized)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    success_count = args.total - args.errors - args.unauthorized
    print(f"Generated {args.total} log lines -> {args.output}")
    print(f"  Success lines:      {success_count}")
    print(f"  Error lines:        {args.errors}")
    print(f"  Unauthorized lines: {args.unauthorized}")


if __name__ == "__main__":
    main()