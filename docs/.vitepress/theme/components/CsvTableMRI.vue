<script setup>
import { ref, onMounted } from 'vue'
import Papa from 'papaparse'

const props = defineProps({
  src: {
    type: String,
    required: true
  }
})

const csvData = ref([])

const loadCsv = async () => {
  try {
    const response = await fetch(props.src)
    const text = await response.text()
    const parsed = Papa.parse(text, { header: false })
    // 过滤空白行（至少有一个非空单元格）
    const filteredData = parsed.data.filter(row =>
      row.some(cell => cell && cell.trim() !== '')
    )
    csvData.value = filteredData
  } catch (error) {
    console.error('加载 CSV 失败:', error)
  }
}

onMounted(() => {
  loadCsv()
})
</script>

<template>
  <div class="csv-table-container">
    <table class="csv-table" v-if="csvData.length">
      <thead>
        <tr>
          <th v-for="(cell, index) in csvData[0]" :key="index">{{ cell }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, rowIndex) in csvData.slice(1)" :key="rowIndex">
          <td v-for="(cell, colIndex) in row" :key="colIndex">{{ cell ? cell.trim() : '' }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else>正在加载数据...</p>
  </div>
</template>

<style scoped>
.csv-table-container {
  max-width: 100%;
  overflow-x: auto;
  padding: 1em;
  background-color: #f9fafb;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.csv-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'Segoe UI', sans-serif;
  font-size: 14px;
  color: #333;
}

.csv-table th {
  background-color: #d6eadf;
  color: #333;
  text-align: left;
  padding: 8px;
  border-bottom: 2px solid #ddd;
}

/* 限制第一列宽度 */
.csv-table th:first-child,
.csv-table td:first-child {
  width: 80px;
  max-width: 80px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.csv-table td {
  padding: 8px;
  border-bottom: 1px solid #e5e7eb;
}

.csv-table tr:hover {
  background-color: #f1f5f9;
}
</style>
