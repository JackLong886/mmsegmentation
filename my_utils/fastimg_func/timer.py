import secrets
import time
from datetime import datetime


class program_progress:
    def __init__(self, print_message: str, stage=1, total_stage=1, if_print_1line=False):
        assert total_stage >= stage
        self.print_message = print_message
        self.stage = stage
        self.total_stage = total_stage
        self.if_subprocess = False
        self.if_ssubprocess = False
        self.if_sssubprocess = False
        self.if_print_1line = if_print_1line
        self.t0 = time.time()
        # print('进度配置成功, 开始时间:{}'.format(get_time()))
        if not self.if_print_1line:
            print('\r{}: {:8f}'.format(self.print_message, (self.stage - 1) / self.total_stage), flush=True)
        else:
            print('\r{}: {:8f}'.format(self.print_message, (self.stage - 1) / self.total_stage), flush=True, end='')

    def __call__(self, v1, v2=None, v3=None):
        # 主进度
        if not self.if_subprocess:
            progress = v1 / self.total_stage + (self.stage - 1) / self.total_stage
        # 三层
        elif self.if_sssubprocess:
            progress = v1 / self.total_stage / self.sub_total_stage / self.ssub_total_stage / self.sssub_total_stage + \
                       (self.stage - 1) / self.total_stage + \
                       (self.sub_stage - 1) / self.sub_total_stage / self.total_stage + \
                       (self.ssub_stage - 1) / self.sub_total_stage / self.total_stage / self.ssub_total_stage + \
                       (
                               self.sssub_stage - 1) / self.sub_total_stage / self.total_stage / self.ssub_total_stage / self.sssub_total_stage
        # 两层
        elif self.if_ssubprocess:
            progress = v1 / self.total_stage / self.sub_total_stage / self.ssub_total_stage + \
                       (self.stage - 1) / self.total_stage + \
                       (self.sub_stage - 1) / self.sub_total_stage / self.total_stage + \
                       (self.ssub_stage - 1) / self.sub_total_stage / self.total_stage / self.ssub_total_stage
        # 一层
        else:
            progress = v1 / self.total_stage / self.sub_total_stage + \
                       (self.stage - 1) / self.total_stage + \
                       (self.sub_stage - 1) / self.sub_total_stage / self.total_stage
        if progress != 0:
            if not self.if_print_1line:
                print('\r{}: {:8f}'.format(self.print_message, progress - 0.0001), flush=True)
            else:
                print('\r{}: {:8f}'.format(self.print_message, progress - 0.0001), flush=True, end='')
        else:
            if not self.if_print_1line:
                print('\r{}: {:8f}'.format(self.print_message, progress), flush=True)
            else:
                print('\r{}: {:8f}'.format(self.print_message, progress), flush=True, end='')

    def __del__(self):
        # self.finish_program()
        if not self.if_print_1line:
            print(f'\rTotal Time: {time.time() - self.t0}')
        else:
            print(f'\rTotal Time: {time.time() - self.t0}')

    def get2print(self, v1):
        # 主进度
        if not self.if_subprocess:
            progress = v1 / self.total_stage + (self.stage - 1) / self.total_stage
        # 三层
        elif self.if_sssubprocess:
            progress = v1 / self.total_stage / self.sub_total_stage / self.ssub_total_stage / self.sssub_total_stage + \
                       (self.stage - 1) / self.total_stage + \
                       (self.sub_stage - 1) / self.sub_total_stage / self.total_stage + \
                       (self.ssub_stage - 1) / self.sub_total_stage / self.total_stage / self.ssub_total_stage + \
                       (
                               self.sssub_stage - 1) / self.sub_total_stage / self.total_stage / self.ssub_total_stage / self.sssub_total_stage
        # 两层
        elif self.if_ssubprocess:
            progress = v1 / self.total_stage / self.sub_total_stage / self.ssub_total_stage + \
                       (self.stage - 1) / self.total_stage + \
                       (self.sub_stage - 1) / self.sub_total_stage / self.total_stage + \
                       (self.ssub_stage - 1) / self.sub_total_stage / self.total_stage / self.ssub_total_stage
        # 一层
        else:
            progress = v1 / self.total_stage / self.sub_total_stage + \
                       (self.stage - 1) / self.total_stage + \
                       (self.sub_stage - 1) / self.sub_total_stage / self.total_stage
        return '\r{}: {:8f}'.format(self.print_message, progress)

    def is_last_stage(self):
        flag = True
        flag = flag and self.total_stage == self.stage
        if self.if_subprocess:
            flag = flag and self.sub_stage == self.sub_total_stage
        if self.if_ssubprocess:
            flag = flag and self.ssub_stage == self.ssub_total_stage
        if self.if_sssubprocess:
            flag = flag and self.sssub_stage == self.sssub_total_stage
        return flag

    def finish_program(self):
        if not self.if_print_1line:
            print('\r{}: {:8f}'.format(self.print_message, self.stage / self.total_stage), flush=True)
        else:
            print('\r{}: {:8f}'.format(self.print_message, self.stage / self.total_stage), flush=True)
        # print("结束时间：{}".format(get_time()))

    def set_subprocess(self, sub_stage: int, sub_total_stage: int):
        # 进入下一层进度条

        if not self.if_subprocess:
            self.if_subprocess = True
        self.sub_stage = sub_stage
        self.sub_total_stage = sub_total_stage

    def set_ssubprocess(self, ssub_stage: int, ssub_total_stage: int):
        if not self.if_ssubprocess:
            self.if_ssubprocess = True
        self.ssub_stage = ssub_stage
        self.ssub_total_stage = ssub_total_stage

    def set_sssubprocess(self, sssub_stage: int, sssub_total_stage: int):
        if not self.if_sssubprocess:
            self.if_sssubprocess = True
        self.sssub_stage = sssub_stage
        self.sssub_total_stage = sssub_total_stage


def get_time():
    now = datetime.now()
    # 使用 strftime() 方法将 datetime 对象转换为字符串
    formatted_time = now.strftime("%Y%m%d%H%M%S%f")
    return formatted_time


def get_time_id():
    return f'{get_time()[2:12]}{secrets.token_hex(2)}'


class Timer:
    def __init__(self, show_message='Total Time'):
        self.start0 = time.time()
        self.start = time.time()
        self.records = {}
        self.total = 0
        self.show_message = show_message

    def elapsed(self):
        end = time.time()
        res = end - self.start
        self.start = end
        return res

    def record(self, category, extra_time=0):
        e = self.elapsed()
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += e + extra_time
        self.total += e + extra_time
        print(f'{category} finish!， {e:.2f}')

    def summary(self):
        res = f"{self.total:.1f}s"

        additions = [x for x in self.records.items() if x[1] >= 0.1]
        if not additions:
            return res

        res += " ("
        res += ", ".join([f"{category}: {time_taken:.1f}s" for category, time_taken in additions])
        res += ")"

        return res

    def reset(self):
        self.__init__()

    def show_time(self):
        print(f"{self.show_message}: {self.summary()}.")

    def __del__(self):
        self.show_time()
        print(f"Real Total Time: {time.time() - self.start0}.")


def timeit(fn):
    # *args and **kwargs are to support positional and named arguments of fn
    def get_cost_time(*args, **kwargs):
        start = time.time()
        output = fn(*args, **kwargs)
        print(f"Time taken in {fn.__name__}: {time.time() - start:.7f}")
        return output  # make sure that the decorator returns the output of fn

    return get_cost_time


def fib_m_helper(n, computed):
    if n in computed:
        return computed[n]
    computed[n] = fib_m_helper(n - 1, computed) + fib_m_helper(n - 2, computed)
    return computed[n]


if __name__ == '__main__':
    # startup_timer = Timer()
    # time.sleep(2)
    # startup_timer.record('wait 2s')
    # time.sleep(1)
    # startup_timer.record('wait 1s')
    # startup_timer.show_time()

    print()
