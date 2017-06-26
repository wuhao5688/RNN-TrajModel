using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.IO;

namespace OSMProcess
{
    class Program
    {
        class Node
        {
            public Node(double lat, double lon)
            {
                this.lat = lat;
                this.lon = lon;
                this.newId = -1;
            }
            public double lat;
            public double lon;
            public int newId;
        }

        class Way
        {
            public List<long> enode = new List<long>();
            public List<bool> mark = new List<bool>();
            public bool isoneway;
            public string wayType;  /////new
        }
        static void Main(string[] args)
        {
            XmlDocument dom = new XmlDocument();
            dom.Load(@"put_the_OSM_raw_data_here.xml");

            Console.WriteLine("xml loaded!");
            StreamWriter edgeSw = new StreamWriter("edgeOSM.txt");
            StreamWriter nodeSw = new StreamWriter("nodeOSM.txt");
            StreamWriter wayTypeSw = new StreamWriter("wayTypeOSM.txt");

            Dictionary<long, Node> nodeDic = new Dictionary<long, Node>();
            List<Node> extractedNodes = new List<Node>();
            List<Way> extractedWays = new List<Way>();
            List<Node> visit = new List<Node>();
            Dictionary<Node, int> split = new Dictionary<Node, int>();
            Dictionary<long, int> newid = new Dictionary<long, int>();
            int nodeCount = 0;
            int wayCount;
            
            //You can comment the road type which you want to included in the road network
            Dictionary<string, bool> waytype = new Dictionary<string, bool>();
            waytype.Add("motorway", true);
            waytype.Add("trunk", true);
            waytype.Add("primary", true);
            waytype.Add("secondary", true);
            waytype.Add("tertiary", true);
            waytype.Add("unclassified", true);
            waytype.Add("residential", true);
            //waytype.Add("service", true); //for example, it is usual to ignore the service roads
            waytype.Add("motorway_link", true);
            waytype.Add("trunk_link", true);
            waytype.Add("primary_link", true);
            waytype.Add("secondary_link", true);
            waytype.Add("tertiary_link", true);

            Dictionary<string, int> waylevel = new Dictionary<string, int>();
            waylevel.Add("motorway", 7);
            waylevel.Add("trunk", 6);
            waylevel.Add("primary", 5);
            waylevel.Add("secondary", 4);
            waylevel.Add("tertiary", 3);
            waylevel.Add("unclassified", 2);
            waylevel.Add("residential", 1);
            waylevel.Add("service", 0);
            waylevel.Add("motorway_link", 7);
            waylevel.Add("trunk_link", 6);
            waylevel.Add("primary_link", 5);
            waylevel.Add("secondary_link", 4);
            waylevel.Add("tertiary_link", 3);

            int icount = 0;
            foreach (XmlElement entry in dom.DocumentElement.ChildNodes)
            {
                if (entry.Name.Equals("node"))
                {
                    long oldId = long.Parse(entry.GetAttribute("id"));
                    double lat = double.Parse(entry.GetAttribute("lat"));
                    double lon = double.Parse(entry.GetAttribute("lon"));
                    Node node = new Node(lat, lon);
                    nodeDic.Add(oldId, node);
                }
                if (entry.Name.Equals("way"))
                {
                    List<long> figure = new List<long>();
                    bool isHighWay = false;
                    bool isOneWay = false;
                    bool isBorder = false;
                    string highwayname = "";
                    foreach (XmlElement childNode in entry.ChildNodes)
                    {
                        if (childNode.Name.Equals("nd"))
                        {
                            figure.Add(long.Parse(childNode.GetAttribute("ref")));
                        }
                        if (childNode.Name.Equals("tag"))
                        {
                            if (childNode.GetAttribute("k").Equals("highway"))
                            {
                                isHighWay = true;
                                highwayname = (childNode.GetAttribute("v"));
                            }
                            if (childNode.GetAttribute("k").Equals("oneway") && childNode.GetAttribute("v").Equals("yes")) //
                                isOneWay = true;
                            if (childNode.GetAttribute("k").Equals("boundary") && childNode.GetAttribute("v").Equals("administrative"))
                                isBorder = true;
                        }
                    }
                    if (isHighWay == false)
                        continue;
                    bool tmp = false;
                    if (!waytype.TryGetValue(highwayname, out tmp))
                    {
                        continue;
                    }
                    if (icount % 10000 == 0)
                        Console.WriteLine(icount);
                    icount++;

                    Way way = new Way();
                    way.enode = figure;
                    way.isoneway = isOneWay;
                    way.wayType = highwayname; //////new 
                    // processing head part
                    Node fromNode;
                    if (nodeDic.TryGetValue(figure[0], out fromNode))
                    {
                        if (fromNode.newId == -1)
                        {
                            newid.Add(figure[0], nodeCount);
                            fromNode.newId = nodeCount++; //id counter
                            extractedNodes.Add(fromNode);
                        }
                        way.mark.Add(true);
                        if (visit.Contains(fromNode))
                        {
                            int edgeId = split[fromNode];
                            if (edgeId != -1 && edgeId < extractedWays.Count)
                            {
                                //split
                                for (int k = 0; k < extractedWays[edgeId].enode.Count; k++)
                                    if (extractedWays[edgeId].enode[k] == figure[0])
                                        extractedWays[edgeId].mark[k] = true;
                                split[fromNode] = -1;
                            }
                        }
                        else
                        {
                            visit.Add(fromNode);
                            split.Add(fromNode,-1);
                        }
                    }
                    else
                    {
                        Console.WriteLine("invalid key!");
                        Console.ReadKey(true);
                    }

                    //
                    for (int i = 1; i < figure.Count - 1; i++)
                    {
                        Node node;
                        if (nodeDic.TryGetValue(figure[i], out node))
                        {
                            if (visit.Contains(node))
                            {
                                int edgeId = split[node];
                                way.mark.Add(true);
                                if (edgeId != -1 && edgeId < extractedWays.Count)
                                {
                                    if (node.newId == -1)
                                    {
                                        newid.Add(figure[i], nodeCount);
                                        node.newId = nodeCount++;
                                        extractedNodes.Add(node);
                                    }
                                    for (int k = 0; k < extractedWays[edgeId].enode.Count; k++)
                                        if (extractedWays[edgeId].enode[k] == figure[i])
                                            extractedWays[edgeId].mark[k] = true;
                                    split[fromNode] = -1;
                                }
                            }
                            else
                            {
                                visit.Add(node);
                                split.Add(node,extractedWays.Count);
                                way.mark.Add(false);
                            }
                        }
                        else
                        {
                            Console.WriteLine("invalid key!");
                            Console.ReadKey(true);
                        }
                    }

                    // process tail part
                    Node toNode;
                    if (nodeDic.TryGetValue(figure[figure.Count - 1], out toNode))
                    {
                        if (toNode.newId == -1)
                        {
                            newid.Add(figure[figure.Count-1], nodeCount);
                            toNode.newId = nodeCount++;
                            extractedNodes.Add(toNode);
                        }
                        way.mark.Add(true);
                        if (visit.Contains(toNode))
                        {
                            int edgeId = split[toNode];
                            if (edgeId != -1 && edgeId < extractedWays.Count)
                            {
                                //split
                                for (int k = 0; k < extractedWays[edgeId].enode.Count; k++)
                                    if (extractedWays[edgeId].enode[k] == figure[figure.Count-1])
                                        extractedWays[edgeId].mark[k] = true;
                                split[toNode] = -1;
                            }
                        }
                        else
                        {
                            visit.Add(toNode);
                            split.Add(toNode, -1);
                        }
                        extractedWays.Add(way);
                    }
                    else
                    {
                        Console.WriteLine("invalid key!");
                        Console.ReadKey(true);
                    }
                }
            }
            //output nodes
            for (int i = 0; i < extractedNodes.Count; i++)
            {
                nodeSw.Write(i + "\t" + extractedNodes[i].lat + "\t" + extractedNodes[i].lon + "\n");
            }
            wayCount = 0;
            for (int i = 0; i < extractedWays.Count; i++)
            {
                int sid = 0;
                for (int j = 1; j < extractedWays[i].enode.Count; j++)
                {
                    if (extractedWays[i].mark[j])
                    {
                        long start = extractedWays[i].enode[sid];
                        long end = extractedWays[i].enode[j];
                        int num = j-sid+1;
                        int startid, endid;
                        if (!newid.TryGetValue(start, out startid))
                            continue;
                        if (!newid.TryGetValue(end, out endid))
                            continue;
                        string str_wayType = extractedWays[i].wayType;
                        int level;
                        waylevel.TryGetValue(str_wayType, out level);
                        wayTypeSw.Write(wayCount + "\t" + str_wayType + "\t" + level + "\n");
                        edgeSw.Write(wayCount+"\t"+startid+"\t"+endid+"\t"+num);
                        for (int k = sid; k <= j; k++)
                        {
                            Node node = nodeDic[extractedWays[i].enode[k]];
                            edgeSw.Write("\t"+node.lat+"\t"+node.lon);
                        }
                        edgeSw.Write("\n");
                        if (!extractedWays[i].isoneway) //output the reversed road if the road is a bi-directional road
                        {
                            wayCount++;    
                            wayTypeSw.Write(wayCount + "\t" + str_wayType + "\t" + level + "\n"); 
                            edgeSw.Write(wayCount + "\t" + endid + "\t" + startid + "\t" + num);
                            for (int k = j; k >= sid; k--)
                            {
                                Node node = nodeDic[extractedWays[i].enode[k]];
                                edgeSw.Write("\t" + node.lat + "\t" + node.lon);
                            }
                            edgeSw.Write("\n");
                        }
                        sid = j;
                        wayCount++;
                    }
                }
            }
            edgeSw.Close();
            nodeSw.Close();
            wayTypeSw.Close();
            Console.WriteLine("road network extraction done！");
           // System("pause");
        }
    }
}