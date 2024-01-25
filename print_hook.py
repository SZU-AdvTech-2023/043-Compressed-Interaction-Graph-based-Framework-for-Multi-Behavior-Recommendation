import sys


# this class gets all output directed to stdout(e.g by print statements)
# and stderr and redirects it to a user defined function
class PrintHook:
    # out = 1 means stdout will be hooked
    # out = 0 means stderr will be hooked
    def __init__(self, out=1):
        self.func = None  ##self.func is userdefined function
        self.origOut = None
        self.out = out

    # user defined hook must return three variables
    # proceed, lineNoMode, newText
    def TestHook(self, text):
        f = open('hook_log.txt', 'a')
        f.write(text)
        f.close()
        return 0, 0, text

    def Start(self, func=None):             #sys.stdout = self: 这行代码将当前的 PrintHook 实例设置为 sys.stdout。在Python中，sys.stdout 通常是指向控制台输出的流，但是这行代码将其更改为指向 PrintHook 的实例。这意味着之后所有尝试打印到控制台的操作（如 print 函数）都会调用 PrintHook 实例的 write 方法。self.origOut = sys.__stdout__: 这行代码保存了原始的 sys.stdout 流，即控制台输出流。sys.__stdout__ 是 sys.stdout 的原始值，即在Python启动时的默认值。通过保存这个值，PrintHook 可以在之后将 sys.stdout 还原回原来的状态，从而结束重定向。简而言之，这两行代码的作用是开始重定向 stdout 输出到 PrintHook 类的实例，同时保存原始的 stdout 流，以便将来能够停止重定向并恢复到正常状态。
        if self.out:
            sys.stdout = self
            self.origOut = sys.__stdout__
        else:
            sys.stderr = self
            self.origOut = sys.__stderr__

        if func:
            self.func = func
        else:
            self.func = self.TestHook

    # Stop will stop routing of print statements thru this class
    def Stop(self):
        self.origOut.flush()
        if self.out:
            sys.stdout = sys.__stdout__
        else:
            sys.stderr = sys.__stderr__
        self.func = None

    # override write of stdout
    def write(self, text):

        bProceed = 1         # 是否继续输出
        bLineNo = 0          # 要不要行号
        newText = ''         # 输出的新内容

        if self.func != None:
            bProceed, bLineNo, newText = self.func(text)

        if bProceed:
            if text.split() == []:
                self.origOut.write(text)
            else:
                # if goint to stdout then only add line no file etc
                # for stderr it is already there
                if self.out:
                    if bLineNo:
                        try:                          
                            raise Exception("Dummy")
                        except:
                            lineNo = 'line(' + str(sys.exc_info()[2].tb_frame.f_back.f_lineno) + '):'
                            codeObject = sys.exc_info()[2].tb_frame.f_back.f_code
                            fileName = codeObject.co_filename
                            funcName = codeObject.co_name
                        self.origOut.write('file ' + fileName + ',' + 'func ' + funcName + ':' + lineNo)
                self.origOut.write(newText)

    # pass all other methods to __stdout__ so that we don't have to override them
    def __getattr__(self, name):
        # return self.origOut.__getattr__(name)
        return getattr(self.origOut, name)


if __name__ == '__main__':
    def MyHookOut(text):
        f = open('log.txt', 'a')
        f.write(text)
        f.close()
        return 1, 1, 'Out Hooked:' + text


    # def MyHookErr(text):
    #     f = open('hook_log.txt', 'a')
    #     f.write(text)
    #     f.close()
    #     return 1, 1, 'Err Hooked:' + text


    print('Hook Start')
    phOut = PrintHook()
    phOut.Start(MyHookOut)
    # phErr = PrintHook(0)               # 传递了一个参数 0，所以 out=0，意味着这个实例将拦截 stderr 的输出。
    # phErr.Start(MyHookErr)
    print('Is this working?')
    print('It seems so!')
    phOut.Stop()
    # print('STDOUT Hook end')
    # compile(',', '<string>', 'exec')
    # phErr.Stop()
    # print('Hook end')


# 异常 (Exception) 是什么？ 在编程中，异常是程序运行时发生的错误，它会中断正常的程序流程。例如，如果你尝试打开一个不存在的文件，程序就会抛出一个异常。raise Exception("Dummy") 是做什么的？ 这行代码故意制造了一个异常情况。raise 关键字用于触发异常。这里创建了一个新的异常实例，其消息是 "Dummy"。这通常是为了测试异常处理逻辑或者在这个例子中，用来获取当前执行的代码位置。为什么要故意制造异常？在这个特定的情况下，我们不是真的关心异常的类型或它的消息（"Dummy"），我们故意触发并立即捕获它，是为了利用 Python 在异常发生时提供的调试信息。 sys.exc_info()[2].tb_frame.f_back.f_lineno 这串代码是怎么工作的？sys.exc_info() 是一个函数，当异常发生时，它会返回包含异常信息的一个元组（tuple）。这个元组有三个元素：异常类型、异常值和一个 traceback 对象。sys.exc_info()[2] 提取这个元组的第三个元素，即 traceback 对象。这个对象包含了异常发生时的详细信息。.tb_frame 获取引发异常的那个代码帧（frame）的对象。.f_back 获取当前帧的前一个帧。这里需要前一个帧是因为当前帧是 raise 语句本身的帧，而我们想知道的是调用 raise 之前的代码位置。.f_lineno 从这个帧对象中获取行号。综上，这串代码的目的是为了获取并构造出一个包含行号的字符串。这个行号指的是调用 print 函数的代码行。这通常用于调试，因为它可以告诉你程序在哪一行出了问题。

# 如果只要写入文件的同时输出到控制台，输出的格式上不做更改，可以用以下代码实现。
# import sys 

# class PrintHook:
#     def __init__(self, filename, out=True):
#         self.func = self.default_hook
#         self.origOut = sys.stdout if out else sys.stderr
#         self.logfile = open(filename, 'a')
#         self.out = out

#     def default_hook(self, text):
#         self.logfile.write(text)
#         self.logfile.flush()
#         self.origOut.write(text)
#         return True, False, text

#     def Start(self):
#         if self.out:
#             sys.stdout = self
#         else:
#             sys.stderr = self

#     def Stop(self):
#         if self.out:
#             sys.stdout = self.origOut
#         else:
#             sys.stderr = self.origOut
#         self.logfile.close()

#     def write(self, text):
#         if self.func:
#             bProceed, bLineNo, newText = self.func(text)
#             if bProceed:
#                 self.origOut.write(newText)
#                 self.logfile.write(newText)
#                 self.logfile.flush()

#     def flush(self):
#         # 这个 flush 方法是需要的，因为 sys.stdout 有一个 flush 方法
#         self.origOut.flush()

# # 使用 PrintHook
# ph = PrintHook('output.txt')
# ph.Start()

# # 下面的 print 会同时在控制台和文件 'output.txt' 中输出
# print("Hello, world!")

# # 当你完成后，停止重定向并关闭文件
# ph.Stop()
