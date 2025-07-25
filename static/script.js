
document.addEventListener('DOMContentLoaded', () => {
    const machineList = document.getElementById('machine-list');
    const machineDetailsTitle = document.getElementById('machine-details-title');
    const machineDetailsContent = document.getElementById('machine-details-content');
    const tempChartCtx = document.getElementById('temp-chart').getContext('2d');
    const vibrationChartCtx = document.getElementById('vibration-chart').getContext('2d');

    let tempChart, vibrationChart;

    const fetchMachines = async () => {
        const response = await fetch('/api/machines');
        const machines = await response.json();
        machineList.innerHTML = '';
        machines.forEach(machine => {
            const li = document.createElement('li');
            li.textContent = machine.machine_id;
            li.dataset.machineId = machine.machine_id;
            li.addEventListener('click', () => fetchMachineDetails(machine.machine_id));
            machineList.appendChild(li);
        });
    };

    const fetchMachineDetails = async (machineId) => {
        const response = await fetch(`/api/machine/${machineId}`);
        const data = await response.json();

        machineDetailsTitle.textContent = `Machine Details: ${machineId}`;
        machineDetailsContent.innerHTML = `
            <p><strong>Type:</strong> ${data[0].machine_type}</p>
            <p><strong>Model:</strong> ${data[0].model}</p>
            <p><strong>Age:</strong> ${data[0].age_years} years</p>
        `;

        updateCharts(data);
    };

    const updateCharts = (data) => {
        const labels = data.map(d => new Date(d.timestamp).toLocaleDateString());
        const tempData = data.map(d => d.temperature_c);
        const vibrationData = data.map(d => d.vibration_mm_s);

        if (tempChart) tempChart.destroy();
        tempChart = new Chart(tempChartCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Temperature (°C)',
                    data: tempData,
                    borderColor: '#ff6384',
                    tension: 0.1
                }]
            }
        });

        if (vibrationChart) vibrationChart.destroy();
        vibrationChart = new Chart(vibrationChartCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Vibration (mm/s)',
                    data: vibrationData,
                    borderColor: '#36a2eb',
                    tension: 0.1
                }]
            }
        });
    };

    fetchMachines();
});
