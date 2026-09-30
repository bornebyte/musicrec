import argparse
from . import db, pipeline


def main():
    ap = argparse.ArgumentParser(prog="musicrec")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    for name in ("extract", "all"):
        p = sub.add_parser(name)
        p.add_argument("--retry", action="store_true", help="retry previously failed files")
        p.add_argument("--rebuild", action="store_true", help="recompute all features")
    sub.add_parser("build"); sub.add_parser("report"); sub.add_parser("eval")
    a = ap.parse_args()
    db.init()
    if a.cmd == "scan": pipeline.scan()
    elif a.cmd == "extract": pipeline.extract(a.retry, a.rebuild)
    elif a.cmd == "build": pipeline.build()
    elif a.cmd == "report": pipeline.report()
    elif a.cmd == "eval": pipeline.evaluate()
    elif a.cmd == "all":
        pipeline.scan(); pipeline.extract(a.retry, a.rebuild); pipeline.build(); pipeline.report()


if __name__ == "__main__":
    main()
