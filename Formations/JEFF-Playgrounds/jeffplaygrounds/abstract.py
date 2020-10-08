import abc


class Base(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def nothing(self):
        """
        Test
        :returns Anything
        """


class AnotherBase(abc.ABC):

    @abc.abstractmethod
    def nothing(self):
        """
        Test
        :returns Anything
        """


try:
    base = Base()
except:
    print("Failed base")

try:
    anotherBase = AnotherBase()
except:
    print("Failed another base")

class BaseImpl(Base):
    def nothing(selfs):
        return "ok"


base = BaseImpl()
print(base.nothing())