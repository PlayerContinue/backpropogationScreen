using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace VisualBackPropogation
{
    /// <summary>
    /// Follow steps 1a or 1b and then 2 to use this custom control in a XAML file.
    ///
    /// Step 1a) Using this custom control in a XAML file that exists in the current project.
    /// Add this XmlNamespace attribute to the root element of the markup file where it is 
    /// to be used:
    ///
    ///     xmlns:MyNamespace="clr-namespace:VisualBackPropogation"
    ///
    ///
    /// Step 1b) Using this custom control in a XAML file that exists in a different project.
    /// Add this XmlNamespace attribute to the root element of the markup file where it is 
    /// to be used:
    ///
    ///     xmlns:MyNamespace="clr-namespace:VisualBackPropogation;assembly=VisualBackPropogation"
    ///
    /// You will also need to add a project reference from the project where the XAML file lives
    /// to this project and Rebuild to avoid compilation errors:
    ///
    ///     Right click on the target project in the Solution Explorer and
    ///     "Add Reference"->"Projects"->[Browse to and select this project]
    ///
    ///
    /// Step 2)
    /// Go ahead and use your control in the XAML file.
    ///
    ///     <MyNamespace:GraphImage/>
    ///
    /// </summary>
    public class GraphImage : Canvas
    {
        //List Containing the Ellipses Which will be drawn as the images
        private List<Ellipse> Nodes = new List<Ellipse>();
        private List<System.Windows.Shapes.Line> Edges = new List<Line>();
        private List<PolyBezierSegment> RecursiveEdges = new List<PolyBezierSegment>();
        private int EllipseDiameter;
        static GraphImage()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(GraphImage), new FrameworkPropertyMetadata(typeof(GraphImage)));
        }


        public void DrawGraph(List<int[]> MapToFrom, int num_nodes)
        {
            int start;
            this.EllipseDiameter = 20;
            if (this.Nodes.Count > num_nodes)
            {
                start = this.Nodes.Count;
                for (int i = 0; i < start - num_nodes; i++)
                {
                    this.Children.Remove(this.Nodes[this.Nodes.Count - 1]);
                    this.Nodes.RemoveAt(this.Nodes.Count - 1);
                }
            }
            else
            {
                start = this.Nodes.Count;
                for (int i = 0; i < num_nodes - start; i++)
                {
                    this.Nodes.Add(AddEllipse(new Ellipse { Width = EllipseDiameter, Height = EllipseDiameter }, 50, i));
                    this.Children.Add(this.Nodes[i]);
                }
            }

            start = this.Edges.Count - 1;
            for (int i = 0; i < MapToFrom.Count; i++)
            {
                if (MapToFrom[i][0] != MapToFrom[i][1])
                {
                    if (i > start)
                    {
                      
                        this.RecursiveEdges.Add(AddNonRecursiveEdge(MapToFrom[i][0], MapToFrom[i][1], new PolyBezierSegment()));
                    }
                    else
                    {
                        this.RecursiveEdges[i] = AddNonRecursiveEdge(MapToFrom[i][0], MapToFrom[i][1], new PolyBezierSegment());
                    }
                   
                }
                else
                {
                    if (i > start)
                    {
                        this.RecursiveEdges.Add(AddRecursiveEdge(MapToFrom[i][0], new PolyBezierSegment()));
                    }
                    else
                    {
                        this.RecursiveEdges[i] = AddRecursiveEdge(MapToFrom[i][0], new PolyBezierSegment());
                    }

                    
                }
            }

            


        }


        private Ellipse AddEllipse(Ellipse Shape, int Radius, int Position)
        {
            if (Position != 0)
            {
                Canvas.SetLeft(Shape, Canvas.GetLeft(this.Nodes[Position - 1]) + Radius + (Radius/2));
                Canvas.SetTop(Shape, Canvas.GetTop(this.Nodes[Position - 1]));
            }
            else
            {
                Canvas.SetLeft(Shape, 45);
                Canvas.SetTop(Shape, 45);
            }
            
            Shape.Stroke = new SolidColorBrush(Colors.Green);
            Shape.Fill = Brushes.Aqua;
            return Shape;
        }

        private PolyBezierSegment AddNonRecursiveEdge(int node1,int node2, PolyBezierSegment TempLine)
        {

            Ellipse LeftNode = Canvas.GetLeft(this.Nodes[node1]) < Canvas.GetLeft(this.Nodes[node2]) ? this.Nodes[node1] : this.Nodes[node2];
            Ellipse RightNode = Canvas.GetLeft(this.Nodes[node1]) >= Canvas.GetLeft(this.Nodes[node2]) ? this.Nodes[node1] : this.Nodes[node2];

            Point StartPoint = new Point(Canvas.GetLeft(LeftNode) + LeftNode.Width/2, Canvas.GetTop(LeftNode));//Point at which the bezel start

            TempLine.Points = new PointCollection() { StartPoint,//Start at middle of left
                new Point(Canvas.GetLeft(LeftNode) + (Canvas.GetLeft(RightNode) - Canvas.GetLeft(LeftNode))/2, Canvas.GetTop(LeftNode) - (LeftNode.Height) - 20), //Go above for a second
                new Point(Canvas.GetLeft(RightNode) + (RightNode.Width/2), Canvas.GetTop(RightNode))//End at middle of right
            };

            // Set up the Path to insert the segments
            PathGeometry path = new PathGeometry();


            PathFigure pathFigure = new PathFigure();
            pathFigure.StartPoint = StartPoint;
            pathFigure.IsClosed = false;
            path.Figures.Add(pathFigure);

            pathFigure.Segments.Add(TempLine);
            System.Windows.Shapes.Path p = new Path();
            p.Stroke = Brushes.Red;
            p.StrokeThickness = 2;
            p.Data = path;
            this.Children.Add(p);



            return TempLine;

        }

        private PolyBezierSegment AddRecursiveEdge(int node, PolyBezierSegment TempLine)
        {

           //BezierSegment curve = new BezierSegment(new Point(11, 11), new Point(22, 22), new Point(15, 15), false);

           Point StartPoint = new Point(Canvas.GetLeft(this.Nodes[node]), Canvas.GetTop(this.Nodes[node]) + this.Nodes[node].Height/2);//Point at which the bezel start

            TempLine.Points = new PointCollection() { StartPoint,//Start at middle of left
                new Point(Canvas.GetLeft(this.Nodes[node]) + (this.Nodes[node].Width/2), Canvas.GetTop(this.Nodes[node]) - (this.Nodes[node].Height) - 20), //Go above for a second
                new Point(Canvas.GetLeft(this.Nodes[node]) + (this.Nodes[node].Width), Canvas.GetTop(this.Nodes[node]) + (this.Nodes[node].Height/2))//End at middle of right
            };

            // Set up the Path to insert the segments
            PathGeometry path = new PathGeometry();
           
            
            PathFigure pathFigure = new PathFigure();
            pathFigure.StartPoint = StartPoint;
            pathFigure.IsClosed = false;
            path.Figures.Add(pathFigure);

            pathFigure.Segments.Add(TempLine);
            System.Windows.Shapes.Path p = new Path();
            p.Stroke = Brushes.Red;
            p.StrokeThickness = 2;
            p.Data = path;
            this.Children.Add(p);
            return TempLine;

        }

        private Line AddEdge(int node1, int node2, int diameter, Line TempLine)
        {

            double X1;
            double X2;
            double Y1;
            double Y2;
            if (Canvas.GetLeft(this.Nodes[node1]) < Canvas.GetLeft(this.Nodes[node2]))//Node 1 is to the left
            {
                X1 = Canvas.GetLeft(this.Nodes[node1]) + diameter;
                X2 = Canvas.GetLeft(this.Nodes[node2]);
            }
            else//Node 2 is to the left
            {
                X1 = Canvas.GetLeft(this.Nodes[node1]);
                X2 = Canvas.GetLeft(this.Nodes[node2]) + diameter;
            }
            
            Y1 = Canvas.GetTop(this.Nodes[node1]) + (diameter / 2);
            Y2 = Canvas.GetTop(this.Nodes[node2]) + (diameter / 2);

            TempLine.X1 = X1;
            TempLine.X2 = X2;
            TempLine.Y1 = Y1;
            TempLine.Y2 = Y2;

            // Create a red Brush
            SolidColorBrush redBrush = new SolidColorBrush();
            redBrush.Color = Colors.Red;

            // Set Line's width and color
            TempLine.StrokeThickness = 4;
            TempLine.Stroke = redBrush;

            return TempLine;
        }
    }
}
