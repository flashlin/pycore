from rx import of, operators as op
from rx.subject import Subject


class RxBus:
    def __init__(self):
        self._bus = Subject()

    def post(self, data):
        self._bus.on_next(data)

    def to_observable(self, event_type_name):
        return self._bus.pipe(
            # op.map(lambda x: type(x).__name__),
            op.filter(lambda x: type(x).__name__ == event_type_name)
        )


def main():
    class CustomEvent:
        def __init__(self):
            super(CustomEvent).__init__()
            self.name = "123"

    bus = RxBus()

    bus.to_observable(CustomEvent.__name__) \
        .subscribe(on_next=lambda x: print(f"received {x.video_name}"),
                   on_error=lambda e: print(f"error {e}"),
                   on_completed=lambda: print("end"))

    evt = CustomEvent()
    bus.post(evt)


if __name__ == '__main__':
    main()
