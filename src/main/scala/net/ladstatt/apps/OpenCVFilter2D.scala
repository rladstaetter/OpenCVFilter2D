package net.ladstatt.apps

import java.io.ByteArrayInputStream
import java.io.File

import scala.collection.mutable.ArrayBuffer
import scala.util.Failure
import scala.util.Success
import scala.util.Try

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc

import javafx.application.Application
import javafx.beans.value.ChangeListener
import javafx.beans.value.ObservableValue
import javafx.collections.ListChangeListener
import javafx.event.Event
import javafx.event.EventHandler
import javafx.geometry.Pos
import javafx.scene.Scene
import javafx.scene.control.ChoiceBox
import javafx.scene.control.ListCell
import javafx.scene.control.ListView
import javafx.scene.control.TextField
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import javafx.scene.layout.GridPane
import javafx.scene.layout.HBox
import javafx.scene.layout.VBox
import javafx.stage.Stage
import javafx.util.Callback

/**
 * For a discussion of the concepts of this application see http://ladstatt.blogspot.com/
 */
trait Utils {
  val runOnMac =
    {
      System.getProperty("os.name").toLowerCase match {
        case "mac os x" => true
        case _ => false
      }
    }

  /**
   * function to measure execution time of first function, optionally executing a display function,
   * returning the time in milliseconds
   */
  def time[A](a: => A, display: Long => Unit = s => ()): A = {
    val now = System.nanoTime
    val result = a
    val micros = (System.nanoTime - now) / 1000
    display(micros)
    result
  }

}

trait OpenCVUtils extends Utils {

  def loadNativeLibs() = {
    val nativeLibName = if (runOnMac) "/opt/local/share/OpenCV/java/libopencv_java244.dylib" else "c:/openCV/build/java/x64/opencv_java244.dll"
    System.load(new File(nativeLibName).getAbsolutePath())
  }

  def filter2D(kernel: Mat)(input: Mat): Mat = {
    val out = new Mat
    Imgproc.filter2D(input, out, -1, kernel)
    out
  }

  def toImage(mat: Mat): Try[Image] =
    try {
      val memory = new MatOfByte
      Highgui.imencode(".png", mat, memory)
      Success(new Image(new ByteArrayInputStream(memory.toArray())))
    } catch {
      case e: Throwable => Failure(e)
    }

}

trait JfxUtils {

  def mkChangeListener[T](onChangeAction: (ObservableValue[_ <: T], T, T) => Unit): ChangeListener[T] = {
    new ChangeListener[T]() {
      override def changed(observable: ObservableValue[_ <: T], oldValue: T, newValue: T) = {
        onChangeAction(observable, oldValue, newValue)
      }
    }
  }
  def mkListChangeListener[E](onChangedAction: ListChangeListener.Change[_ <: E] => Unit) = new ListChangeListener[E] {
    def onChanged(changeItem: ListChangeListener.Change[_ <: E]): Unit = {
      onChangedAction(changeItem)
    }
  }

  def mkCellFactoryCallback[T](listCellGenerator: ListView[T] => ListCell[T]) = new Callback[ListView[T], ListCell[T]]() {
    override def call(list: ListView[T]): ListCell[T] = listCellGenerator(list)
  }

  def mkEventHandler[E <: Event](f: E => Unit) = new EventHandler[E] { def handle(e: E) = f(e) }

}
/**
 * a variable sized gridpane, constrained by size
 */
class KernelInputArray(size: Int, kernelData: ArrayBuffer[Float], applyKernel: => Mat => Unit) extends GridPane with JfxUtils with OpenCVUtils {

  def mkKernel = {
    val kernel = new Mat(size, size, CvType.CV_32FC1)
    kernel.put(0, 0, kernelData.toArray)
    kernel
  }

  for {
    row <- 0 until size
    col <- 0 until size
  } yield {
    val textField = {
      val tf = new TextField
      tf.setPrefWidth(50)
      tf.setText(kernelData(row * size + col).toString)
      tf.textProperty().addListener(
        mkChangeListener[String](
          (obVal, oldVal, newVal) => {
            try {
              kernelData(row * size + col) = tf.getText.toFloat
              applyKernel(mkKernel)
            } catch {
              case e => // no float, ignore
            }
          }))
      tf
    }
    GridPane.setRowIndex(textField, row)
    GridPane.setColumnIndex(textField, col)
    getChildren.add(textField)
  }
  applyKernel(mkKernel)

}

object OpenCVFilter2D {

  def main(args: Array[String]): Unit = {
    Application.launch(classOf[OpenCVFilter2D], args: _*)
  }

}

class OpenCVFilter2D extends Application with JfxUtils with OpenCVUtils {

  override def init(): Unit = loadNativeLibs // important to have this statement on the "right" thread

  def readImage(file: File): Mat = Highgui.imread(file.getAbsolutePath()) // , 0)

  /**
   * a map of predefined kernels which can serve as starting points for your experiments
   */
  def kernels: Map[String, (Int, ArrayBuffer[Float], Float, Float)] = Map(
    "unit" -> (3, ArrayBuffer[Float](
      0, 0, 0,
      0, 1, 0,
      0, 0, 0), 1, 0),
    "blur" -> (3, ArrayBuffer[Float](
      0, 0.2f, 0,
      0.2f, 0.2f, 0.2f,
      0, 0.2f, 0), 1, 0),
    "moreblur" -> (5, ArrayBuffer[Float](
      0, 0, 1, 0, 0,
      0, 1, 1, 1, 0,
      1, 1, 1, 1, 1,
      0, 1, 1, 1, 0,
      0, 0, 1, 0, 0), 1f / 13, 0),
    "motionblur" -> (9, ArrayBuffer[Float](
      1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1), 1f / 9, 0),
    "horizontaledges" -> (5, ArrayBuffer[Float](
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      -1, -1, 2, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0), 1, 0),
    "verticaledges" -> (5, ArrayBuffer[Float](
      0, 0, -1, 0, 0,
      0, 0, -1, 0, 0,
      0, 0, 4, 0, 0,
      0, 0, -1, 0, 0,
      0, 0, -1, 0, 0), 1, 0),
    "diagonaledges" -> (5, ArrayBuffer[Float](
      -1, 0, 0, 0, 0,
      0, -2, 0, 0, 0,
      0, 0, 6, 0, 0,
      0, 0, 0, -2, 0,
      0, 0, 0, 0, -1), 1, 0),
    "alledges" -> (3, ArrayBuffer[Float](
      -1, -1, -1,
      -1, 8, -1,
      -1, -1, -1), 1, 0),
    "sharpen" -> (3, ArrayBuffer[Float](
      -1, -1, -1,
      -1, 9, -1,
      -1, -1, -1), 1, 0),
    "subtlesharpen" -> (5, ArrayBuffer[Float](
      -1, -1, -1, -1, -1,
      -1, 2, 2, 2, -1,
      -1, 2, 8, 2, -1,
      -1, 2, 2, 2, -1,
      -1, -1, -1, -1, -1), 1f / 8, 0),
    "excesssharpen" -> (3, ArrayBuffer[Float](
      1, 1, 1,
      1, -7, 1,
      1, 1, 1), 1, 0),
    "emboss" -> (3, ArrayBuffer[Float](
      -1, -1, 0,
      -1, 0, 1,
      0, 1, 1), 1f, 0.1f),
    "emboss2" -> (5, ArrayBuffer[Float](-1, -1, -1, -1, 0,
      -1, -1, -1, 0, 1,
      -1, -1, 0, 1, 1,
      -1, 0, 1, 1, 1,
      0, 1, 1, 1, 1), 1, 0))

  def mkKernelInputArray(initialKernelName: String, mutateFn: => Mat => Unit): KernelInputArray = {
    val (size, kernelData, factor, bias) = kernels(initialKernelName)
    val initalKernel = kernelData.map(_ * factor + bias)
    new KernelInputArray(size, initalKernel, mutateFn)
  }

  override def start(stage: Stage): Unit = {
    stage.setTitle("2D Image Filters with OpenCV and JavaFX")
    val canvas = new HBox
    canvas.setAlignment(Pos.CENTER)
    val choiceBox = new ChoiceBox[String]
    choiceBox.getItems.addAll(kernels.keySet.toSeq.sortWith(_ < _): _*)
    choiceBox.setValue("unit")
    val input = readImage(new File("src/main/resources/turbine.png"))

    toImage(input) match {
      case Failure(e) => println(e.getMessage())
      case Success(inputImage) => {

        def mutateOutputImage(outputView: ImageView)(kernel: Mat): Unit = {
          toImage(filter2D(kernel)(input)) match {
            case Failure(e) =>
            case Success(mutatedImage) => outputView.setImage(mutatedImage)
          }
        }

        val originalView = new ImageView(inputImage)
        val outputView = new ImageView(inputImage) // show input image initially

        choiceBox.valueProperty().addListener(mkChangeListener[String](
          (obVal, oldVal, newVal) => {
            val cell = mkKernelInputArray(newVal, mutateOutputImage(outputView))
            canvas.getChildren().clear()
            val kernelAndChoiceBox = new VBox
            kernelAndChoiceBox.setAlignment(Pos.CENTER)
            kernelAndChoiceBox.getChildren.addAll(choiceBox, cell)
            canvas.getChildren().addAll(originalView, kernelAndChoiceBox, outputView)
          }))

        val kernelAndChoiceBox = new VBox
        kernelAndChoiceBox.setAlignment(Pos.CENTER)
        kernelAndChoiceBox.getChildren.addAll(choiceBox, mkKernelInputArray("unit", mutateOutputImage(outputView)))

        canvas.getChildren().addAll(originalView, kernelAndChoiceBox, outputView)
      }
    }

    val scene = new Scene(canvas)
    stage.setScene(scene)
    stage.show
  }
}
