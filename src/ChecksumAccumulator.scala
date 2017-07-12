/**
  * Created by qiniu on 2017/7/11.
  */
import scala.collection.mutable.Map

object ChecksumAccumulator {
  private val cache = Map[String, Int]()

  def main(args: Array[String]): Unit = {
    println(ChecksumAccumulator.calculate("Every value is an object"))
  }

  def calculate(s:String): Int =
    if (cache.contains(s))
      cache(s)
    else {
      val acc = new ChecksumAccumulator
      for (c <- s)
        acc.add(c.toByte)
      val cs = acc.checksum()
      cache += (s -> cs)
      cs
    }


}

class ChecksumAccumulator {
  private var sum = 0

  def test(): Unit = println(sum)
  def add(b: Byte) { sum += b}
  def test2() { "1111" }
  def test3() = {"2222"}
  def checksum(): Int = ~(sum & 0xFF) + 1


}

