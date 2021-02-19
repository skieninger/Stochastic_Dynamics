"""Run tests and analyse coverage
Requires pytest, pytest-cov and anybadge
"""

from html.parser import HTMLParser
import shlex
import subprocess


class TotalCoverageParser(HTMLParser):
    """Hacky HTML parser
    Find total code coverage in pytest-cov output
    """
    def __init__(self):
        super().__init__()
        self.found = False
        self.coverage_value = 0

    def handle_starttag(self, tag, attrs):
        for attr in attrs:
            if attr == ("class", "total"):
                self.found = True
                continue

    def handle_data(self, data):
        if not self.found:
            return

        if "%" in data:
            self.coverage_value = data.rstrip("%")
            self.found = False


if __name__ == "__main__":
    test_command = "pytest --cov-report html --cov=Code"
    subprocess.run(shlex.split(test_command))

    with open("htmlcov/index.html") as f_:
        coverage_content = f_.read()

    parser = TotalCoverageParser()
    parser.feed(coverage_content)

    gen_badge_command = (
        f"anybadge --value={parser.coverage_value} "
        f"--file=Tests/coverage_badge.svg --overwrite coverage"
    )
    subprocess.run(shlex.split(gen_badge_command))
