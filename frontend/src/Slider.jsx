export default function Slider({label,min,max,step,value,onChange}) {
    return (
      <label className="block">
        <span className="text-sm">{label}: <b>{value}</b></span>
        <input type="range" min={min} max={max} step={step}
               value={value}
               onChange={e=>onChange(parseFloat(e.target.value))}
               className="w-full"/>
      </label>
    );
  }