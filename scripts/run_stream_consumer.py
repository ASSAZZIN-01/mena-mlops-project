"""Run the Redis Streams inference consumer."""

from mena_mlops.inference.streaming import create_consumer


def main() -> None:
    consumer = create_consumer()
    consumer.ensure_group()
    while True:
        consumer.run_once()


if __name__ == "__main__":
    main()
