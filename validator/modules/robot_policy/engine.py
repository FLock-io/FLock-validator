import hashlib,json,math,random,time
from pathlib import Path
from dataclasses import dataclass,asdict
import yaml

def hobj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def hfile(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

@dataclass
class Episode:
    scenario_id:str; success:bool; completion_time_s:float|None; stable_fraction:float|None; evidence:dict
    def dict(self):return asdict(self)

class Manifest:
    def __init__(self,path):
        self.path=Path(path)
        self.raw=yaml.safe_load(self.path.read_text())
    def resolve(self,p):return (self.path.parent/p).resolve()
    @property
    def hash(self):return hobj(self.raw)
    @property
    def policy(self):return (self.path.parent/self.raw["model"]["path"]).absolute()
    @property
    def policy_hash(self):
        main = hfile(self.policy)
        external = self.raw["model"].get("external_files") or []
        if not external:
            return main
        return hobj({"model": main, "external_files": {
            name: hfile(self.policy.parent / name) for name in sorted(external)}})

class Suite:
    def __init__(self,m):
        self.path=m.resolve(m.raw["suite"]["path"]);self.cfg=yaml.safe_load(self.path.read_text())
        if self.cfg["suite_id"]!=m.raw["suite"]["id"]:raise ValueError("suite id mismatch")
    @property
    def family(self):
        # Two suites with the same family are independent DRAWS from the same
        # distribution -- aggregating across them is meaningful. Two suites with
        # different families are different benchmarks and must not be pooled.
        return self.cfg.get("suite_family") or self.cfg["suite_id"]
    @property
    def hash(self):return hobj(self.cfg)
    def cases(self):
        r=random.Random(self.cfg["suite_seed"]);d=self.cfg["scenario_distribution"]
        for i in range(self.cfg["episodes"]):
            q={}
            for k,v in d.get("perturbations",{}).items():
                q[k]=r.choice(v) if isinstance(v,list) else r.uniform(v["min"],v["max"]) if isinstance(v,dict) and "min" in v else v
            yield {"id":f'{self.cfg["suite_id"]}-{i+1:06d}',"seed":r.randrange(2**31-1),"initial":r.choice(d["initial_states"]),"timeout":self.cfg["timeout_s"],"perturbations":q}

def wilson(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;hh=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [max(0,c-hh),min(1,c+hh)]

def _progress(path, payload):
    """Write a one-line progress snapshot, atomically.

    A 100-episode physics run takes ~7 minutes and printed nothing until it
    finished, so any UI following it had to guess. Written via replace() so a
    reader never sees a half-written file.
    """
    if not path:
        return
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def chunks(cases,batch):
    """Consecutive runs of cases sharing (initial, timeout), at most `batch` long."""
    out=[];cur=[]
    for c in cases:
        key=(c.get("initial"),round(float(c.get("timeout",0)),6))
        if cur and (len(cur)>=batch or (cur[0].get("initial"),round(float(cur[0].get("timeout",0)),6))!=key):
            out.append(cur);cur=[]
        cur.append(c)
    if cur:out.append(cur)
    return out

def run(manifest_path,validator,runner,max_episodes=None,progress_path=None):
    from .adapters import get_adapter
    m=Manifest(manifest_path);s=Suite(m);cfg=yaml.safe_load(m.resolve(m.raw["task"]["config"]).read_text())
    adapter=get_adapter(cfg);eps=[];checked=False
    if type(s.cfg.get("episodes")) is not int or s.cfg["episodes"] <= 0:
        raise ValueError("suite episodes must be a positive integer")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    total=min(s.cfg["episodes"],max_episodes) if max_episodes else s.cfg["episodes"]
    t0=time.time()
    _progress(progress_path,{"validator":validator,"done":0,"total":total,
                             "successes":0,"status":"starting",
                             "suite":s.cfg["suite_id"],"started":t0})
    cases=list(s.cases())
    if max_episodes is not None:cases=cases[:max_episodes]
    def finish(c,tr,ev):
        nonlocal checked
        if not tr:raise RuntimeError(f"empty trajectory for scenario {c['id']}")
        if not checked:
            missing=[k for k in adapter.required_fields if k not in tr[0]]
            if missing:raise RuntimeError(
                f"{type(runner).__name__} does not emit {missing} required by the "
                f"{adapter.family!r} adapter; it emits {sorted(tr[0])}")
            checked=True
        ok,t,st=adapter.evaluate(tr,cfg);eps.append(Episode(c["id"],ok,t,st,ev))
        k=sum(e.success for e in eps)
        # The progress file carried pass/fail and nothing else, so a live view
        # of a run could only ever plot a step function. stable_fraction is the
        # CONTINUOUS signal the adapter already computed -- how much of the
        # episode was within tolerance -- and it separates policies that a
        # binary rate reports identically. The confidence interval goes out
        # too, because a rate without one is not a measurement yet.
        _progress(progress_path,{"validator":validator,"done":len(eps),"total":total,
                                 "successes":k,"rate":k/len(eps),"status":"running",
                                 "ci95":wilson(k,len(eps)),
                                 "suite":s.cfg["suite_id"],"started":t0,
                                 "elapsed":time.time()-t0,
                                 "last_scenario":c["id"],
                                 "last_success":bool(ok),
                                 "last_stable":st,
                                 "mean_stable":sum(e.stable_fraction for e in eps)/len(eps)})
    batch=int(getattr(runner,"batch",1) or 1)
    if batch>1 and hasattr(runner,"rollout_batch"):
        # N scenarios as N parallel envs (DuckEMW's evaluation-battery idea).
        # Chunks never mix initial state or timeout: those are env-wide.
        for chunk in chunks(cases,batch):
            for c,(tr,ev) in zip(chunk,runner.rollout_batch(chunk,m),strict=True):finish(c,tr,ev)
    else:
        # max_episodes truncates the FROZEN suite -- smoke runs only. A result
        # produced this way is not comparable to a full-suite benchmark claim,
        # so it is recorded in the metadata below.
        for c in cases:
            tr,ev=runner.rollout(c,m);finish(c,tr,ev)
    n=len(eps);k=sum(e.success for e in eps)
    return {"metadata":{"validator_id":validator,"manifest_hash":m.hash,"policy_sha256":m.policy_hash,"suite_hash":s.hash,"task_config_hash":hobj(cfg),"suite_id":s.cfg["suite_id"],"suite_family":s.family,**getattr(runner,"describe",lambda:{"substrate":getattr(runner,"substrate","simulation")})(),"partial_suite":max_episodes is not None,"episodes_requested":s.cfg["episodes"],"embodiment":m.raw["embodiment"],"task":m.raw["task"],"task_family":adapter.family,"simulator":m.raw["simulator"]},"metrics":{"episodes":n,"successes":k,"task_success_rate":k/n,"task_success_ci95":wilson(k,n),**adapter.metrics(eps,s.cfg.get("timeout_s"))},"episodes":[e.dict() for e in eps]}
