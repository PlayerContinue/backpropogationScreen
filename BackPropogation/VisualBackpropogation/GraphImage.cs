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
        static GraphImage()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(GraphImage), new FrameworkPropertyMetadata(typeof(GraphImage)));
        }


        public void DrawGraph(List<int[]> MapToFrom, int num_nodes)
        {
            int start;

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
                    this.Nodes.Add(AddEllipse(new Ellipse{Width = 10, Height = 10}, 50, i));
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
                        this.Edges.Add(AddEdge(MapToFrom[i][0], MapToFrom[i][1], new Line()));
                        this.Children.Add(this.Edges.Last<Line>());
                    }
                    else
                    {
                        this.Edges[i] = AddEdge(MapToFrom[i][0], MapToFrom[i][1], new Line());
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

        private PolyBezierSegment AddRecursiveEdge(int node, PolyBezierSegment TempLine)
        {
            TempLine.Points = new PointCollection() { new Point(Canvas.GetLeft(this.Nodes[node]), Canvas.GetTop(this.Nodes[node])),
                new Point(Canvas.GetLeft(this.Nodes[node])+ (this.Nodes[node].Width/2), Canvas.GetTop(this.Nodes[node]) + (this.Nodes[node].Height)+5),
                new Point(Canvas.GetLeft(this.Nodes[node]) + (this.Nodes[node].Width), Canvas.GetTop(this.Nodes[node]) + (this.Nodes[node].Height/2))
            };

            return TempLine;

        }

        private Line AddEdge(int node1, int node2, Line TempLine)
        {
            double X1 = Canvas.GetLeft(this.Nodes[node1]);
            double X2 = Canvas.GetLeft(this.Nodes[node2]);
            double Y1 = Canvas.GetTop(this.Nodes[node1]);
            double Y2 = Canvas.GetTop(this.Nodes[node2]);

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
