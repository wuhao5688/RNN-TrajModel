double MapMatcher::cal_logEmiProb(GeoPoint* p, int s)
{
	//////////////////////////////////////////////////////////////////////////
	/// Compute the emission probability like the GIS'09 paper
	/// p: a observation
	/// s: a candidate state, i.e., a road id
	//////////////////////////////////////////////////////////////////////////
	double dist = roadNet->dist(p->lat, p->lon, roadNet->edges[s]); //compute the vertical distance from a point to an edge, in meter, should be implemented by yourself
	return -0.5 * ((dist - mu) / sigma) * ((dist - mu) / sigma);
}

double MapMatcher::cal_logTransProb(GeoPoint* p_pre, GeoPoint* p, int s1, int s2)
{
	//////////////////////////////////////////////////////////////////////////
	/// Compute the transition probability like the GIS'09 paper
	/// p_pre: the previous observation
	/// p: the current observation
	/// s1: the candidate state of p_pre, a road id
	/// s2: the candidate state of p, a road id
	//////////////////////////////////////////////////////////////////////////
	Edge* e1 = roadNet->edges[s1], *e2 = roadNet->edges[s2];
	if (e1 == NULL || e2 == NULL)
	{
		cout << "Error@MapMatcher::cal_logTransProb: edge is NULL" << endl;
		system("pause");
		exit(0);
	}
	double dist_G = 0.0;
	if (s1 == s2)
	{
		double start2p1, start2p2; // the road network distance from the start node of e1/e2 to the projection of p1/p2 on e1/e2
		start2p1 = roadNet->start2projection(p_pre->lat, p_pre->lon, e1); //compute the road network distance from the start node of an edge to the projection of a given point onto a given edge, should be implemented by yourself
		start2p2 = roadNet->start2projection(p->lat, p->lon, e1);
		if (start2p2 >= start2p1)
			dist_G += (start2p2 - start2p1); //p1->p2 is along direction of the road
		else
			dist_G += (start2p1 - start2p2); //p2->p1 is along direction of the road
	}
	else
	{
		double e1start2p1, e2start2p2; // the road network distance from the start node of e1/e2 to the projection of p1/p2 on e1/e2
		e1start2p1 = roadNet->start2projection(p_pre->lat, p_pre->lon, e1);
		e2start2p2 = roadNet->start2projection(p->lat, p->lon, e2);
		dist_G += (e1->lengthM - e1start2p1 + e2start2p2);
		if (e1->endNodeId != e2->startNodeId)
		{
			dist_G += roadNet->cal_SP(e1->endNodeId, e2->startNodeId); // compute the shortest path given two nodes in the road network, should be implemented by yourself
		}		
	}
	double dist = GeoPoint::distM(p_pre, p);
	return -fabs(dist - dist_G) / beta;
}


bool MapMatcher::MapMatching_kernel(list<GeoPoint*>& traj, double candidateRangeM)
{
	//////////////////////////////////////////////////////////////////////////
	/// Map matching for a trajectory with HMM
	/// traj: here refers to the point trajectory (raw trajectory), which is a list of points/coordinates.
	/// candidateRangeM: the range to find candidate road segments, in meter, for GPS, 50m is enough
	/// Ensure that you have dropped out all the points which have no road nearby within `candidateRangeM` meters.
	///
	/// Note: things you have to implement by your own
	/// - Design a `GeoPoint` class which has the field `lat` and `lon` (should be already converted into the rectangular coordinate) as well as `mmRoadId`, which will be overwritten by the answer after calling this function.
	/// - Design a `RoadNet` class which 
	///		- support indexing edge given an edge id, i.e., `roadNet->edges[]`
	///		- support range query given a point and a range, i.e., `roadNet->getNearEdges_s()`
	///		- support compute the vertical distance form a given point to a given edge, i.e., `roadNet->dist()`
	///     - support compute the road network distance from a given edge's start node to a given point's projection onto the given edge, i.e. `roadNet->start2projection()`
	///		- support shortest path algorithm which will be used in computing transition probability as well as connecting inadjacent road segments to form a vaild route, i.e., `roadNet->calSP()`
	/// If you want to get the legal route (i.e., a edge sequence with each edge being adjacent), just insert the shortest path into the edges which are not adjacent, to form a continuous route
	//////////////////////////////////////////////////////////////////////////

	int maxEdgeNum = roadNet->edges.size(); // #edges in a road network
	int seqLen = traj.size(); // #points in a given trajectory
	vector<pair<int, double> > logProbPrev; //records in the previous time step, if the state is `logProbPrev[i].first`, its corresponding log probability is `logProbPrev[t].second`
	vector<vector<pair<int, int> > > pathMat; //records in time step t, if the state is `pathMat[t][i].first`, its corresponding best state in t-1 is `pathMat[t-1][pathMat[t][i].second].first`
	list<GeoPoint*>::iterator ptIter = traj.begin();
	list<GeoPoint*>::iterator pre_ptIter = traj.begin();
	list<vector<Edge*>>::iterator candidates_traj_iter = candidates_traj.begin();
	bool firstFlag = true;
	int t = 0; //record for time step
	
	while (ptIter != traj.end())
	{
		GeoPoint* pt = (*ptIter);
		GeoPoint* pt_pre = (*pre_ptIter);

		vector<Edge*> candidates;
		roadNet->getNearEdges_s(pt->lat, pt->lon, (double)candidateRangeM, candidates); //find the candidate roads given `pt`, should be implemented by yourself
		if (candidates.size() == 0)
		{
			cout << "Please drop out those points having no candidate within the given range" << endl;
			return false;
		}

		vector<pair<int, double> > logProbCurrent;
		vector<pair<int, int> > currentStates_for_pathMat;
		for each (Edge* edge in candidates)
		{
			int s = edge->id;
			double logEmiProb = cal_logEmiProb(pt, s); //compute the log emission probability, should be implemented by yourself
			currentStates_for_pathMat.push_back(make_pair(s, -1));
			if (firstFlag)
			{
				//initialize logProbCurrent[i].second = P(s|p_0)， logProbCurrent[i].first = s
				logProbCurrent.push_back(make_pair(s, logEmiProb));						
			}
			else
			{				
				//M[i, s] = max{M[i-1, s'] + P(s|s') + P(o|s)} where M[i-1, s'] > -INF
				double maxProb = -INF;
				for (int i = 0; i < logProbPrev.size(); i++)
				{
					int s_prev = logProbPrev[i].first;
					if (roadNet->edges[s_prev] == NULL)
					{
						cout << "error: edge is null" << endl;
						return false;
					}						
					double logTransProb = cal_logTransProb(pt_pre, pt, s_prev, s, use_prune); // compute the log transition probability, should be implemented by yourself
					if (logProbPrev[i].second + logTransProb + logEmiProb > maxProb)
					{
						maxProb = logProbPrev[i].second + logTransProb + logEmiProb;
						currentStates_for_pathMat.back().second = i; //save path
					}
				}
				if (maxProb > -INF)
				{
					logProbCurrent.push_back(make_pair(s, maxProb));
				}
				else
				{
					printf("maxProb is INF\n");
					return false;
				}
			} // END: if (firstFlag) else
		} //END for each (Edge* edge in candidates)
		logProbPrev = logProbCurrent;
		pathMat.push_back(currentStates_for_pathMat);
		//update iterate info
		if (firstFlag)
			firstFlag = false;
		else
			pre_ptIter++;
		ptIter++;
		if (candidates_traj.size() == traj.size())
			candidates_traj_iter++;
		t++;
	} //END: while (ptIter != traj.end())
	
	//retrieve the path
	//find the state of the last time step having the highest log prob
	double maxProb = -INF;
	int last_state = -1;
	for (int i = 0; i < logProbPrev.size();
	int last_idx = -1; i++)
	{
		if (logProbPrev[i].second > maxProb)
		{
			maxProb = logProbPrev[i].second;
			last_state = logProbPrev[i].first;
			last_idx = i;
		}
	}
	//then go back till the head
	vector<int> rev_path;
	int succ_state = last_state; //record the best state of time step t+1
	int succ_idx = last_idx; //record the position of `succ_state` in pathMat[t+1]
	ptIter = traj.end(); ptIter--;
	for (t = traj.size()-1; t >=0; ptIter--, t--)
	{
		if (t == traj.size() - 1)
		{
			(*ptIter)->mmRoadId = last_state; //`mmRoadId` now records the matched road segment, just implement your GeoPoint class by your own.
		}
		else
		{
			int cur_idx = pathMat[t + 1][succ_idx].second;
			int cur_state = pathMat[t][cur_idx].first;
			(*ptIter)->mmRoadId = cur_state;
			succ_idx = cur_idx;
		}
	}
	return true;
}
