/**
  * Created by qiniu on 2017/7/11.
  */
import java.io._
import scala.io.Source

object writeFile {
  def main(args: Array[String]): Unit = {

    val check = new ChecksumAccumulator()
    //val f = new operation_file()
    //f.write_file()
    //f.read_file()
    //f.reduce_list()
    check.test()
    check.add(1)
    check.checksum()
    println(check.test3().isInstanceOf[String])
    println(check.test3().isInstanceOf[Unit])
    println(check.test2().isInstanceOf[Unit])

  }

  class operation_file
  {
    private val path = "write.txt";println("this path is " + path)

    def write_file(): Unit =
    {
      val file = new PrintWriter(new File(path))
      var test_str = Array("a", "b", "c")
      test_str.foreach(file.println)

    }

    def read_file(): Unit = {
      Source.fromFile(path).getLines().foreach(println)
      val list_file = Source.fromFile(path).getLines().toList
      list_file.foreach(println)

    }

    def reduce_list(): Unit = {
      val someint = Array(10, 10, 10)
      val maxint = Array(1,2,3)
      val minint = Array(3, 2, 1 )
      println(someint.reduceLeft(_+_))
      println(someint.reduceLeft(_*_))
      println(someint.reduceLeft(_ max _))
      val f = (x:Int, y:Int) => if (x > y) x else y
      println(maxint.reduceLeft(f))
      val f2 = (x:Int, y:Int) => if (x<y) x else y
      println(maxint.reduceLeft(f2))
    }


  }





}
